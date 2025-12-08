import sys
import time
import serial

from dm2c import ModbusRTU
from lms_drive import LMS_Driver, LMS

# ==================== 基础测试函数（原有部分，略作完善） ====================

# Modbus 串口参数（按你现在的环境来）
DEFAULT_PORT = "COM8"
DEFAULT_BAUDRATE = 115200
DEFAULT_SLAVE_ADDR = b"\x02"  # 你之前设置的地址 02


def init_lms():
    """
    打开串口 & 创建驱动对象（只做连接，不做初始化流程）
    """
    ser = serial.Serial(DEFAULT_PORT, DEFAULT_BAUDRATE, timeout=0.1)
    modbus = ModbusRTU(ser)
    drv = LMS_Driver(modbus, DEFAULT_SLAVE_ADDR)
    return drv


def reset(drv: LMS_Driver):
    """
    伺服的基本初始化：
      - 读取最大加减速度限制
      - 把轮廓加/减速度和 QuickStop 减速度拉到这个上限
      - 初始化限位输入
      - 设置 CiA402 + PP 模式
      - 使能电机
    """
    # 1. 读最大加/减速度限制
    amax_hex = drv.read(LMS.MaxAccelerationLimit, 2)
    dmax_hex = drv.read(LMS.MaxDecelerationLimit, 2)
    try:
        amax = int(amax_hex, 16) if amax_hex else 0
        dmax = int(dmax_hex, 16) if dmax_hex else 0
    except ValueError:
        amax = dmax = 0

    if amax and dmax:
        # 直接写入 6083h / 6084h / 6085h
        drv.write(LMS.ProfileAcceleration, "%08X" % amax)
        drv.write(LMS.ProfileDeceleration, "%08X" % dmax)
        drv.write(LMS.QuickStopDeceleration, "%08X" % dmax)

    # 2. 设置限位
    drv.init_limit_inputs()

    # 3. 切换模式 & 使能
    drv.initPPMode()
    if drv.isFault():          # 有故障先复位
        drv.faultReset()
    drv.enableOperation()      # 6040h = 0x06 -> 0x07 -> 0x0F


def runRelativePosition(drv: LMS_Driver, distance: int):
    """
    以 PP 模式做一个相对位移。
    这里简单读一下最大加减速度，用上限跑。
    """
    amax_hex = drv.read(LMS.MaxAccelerationLimit, 2)
    dmax_hex = drv.read(LMS.MaxDecelerationLimit, 2)
    try:
        amax = int(amax_hex, 16) if amax_hex else 0
        dmax = int(dmax_hex, 16) if dmax_hex else 0
    except ValueError:
        amax = dmax = 0

    if not amax:
        amax = 1000000
    if not dmax:
        dmax = 1000000

    # 速度这里随便给一个比较安全的值
    velocity = 500000

    drv.movePP(
        target_pos=distance,
        velocity=velocity,
        accel=amax,
        decel=dmax,
        relative=True,
        immediate=True,
        wait=False,
    )


def FR(drv: LMS_Driver):
    """
    故障复位：调用 faultReset，然后尝试重新使能。
    """
    drv.faultReset()
    time.sleep(0.1)
    if not drv.isFault():
        try:
            drv.enableOperation()
        except Exception:
            pass


# ==================== 报警代码解析（来自说明书表 8-1） ====================

ERROR_CODE_DESCRIPTIONS = {
    0x2300: "电机过流",
    0x2311: "电机过载",
    0x2312: "电机堵转/堵转并锁轴",
    0x3210: "电源过电压",
    0x3220: "电源欠电压",
    0x4210: "温度过高报警",
    0x4220: "温度过低报警",
    0x5080: "驱动器故障",
    0x5540: "Flash 操作故障",
    0x5541: "Flash 初始化故障",
    0x5542: "Flash 校验错误报警",
    0x5543: "Flash 用户区无参数",
    0x5544: "掉电位置存储异常",
    0x5545: "掉电保存数据未存储",
    0x6000: "硬件初始化故障",
    0x6320: "参数设置错误",
    0x6321: "注册故障",
    0x7301: "编码器电压低故障",
    0x7302: "编码器电压欠压报警",
    0x7303: "编码器通讯故障",
    0x7304: "编码器多圈计数错误报警",
    0x7305: "Z 脉冲故障",
    0x7306: "编码器故障",
    0x7307: "编码器报警",
    0x7309: "编码器内部故障",
    0x7310: "超速",
    0x7501: "Modbus 通信中的非法功能",
    0x7502: "Modbus 通信中的非法数据地址",
    0x7503: "Modbus 通信中的非法数据值",
    0x7505: "Modbus 通信中的确认错误",
    0x7506: "Modbus 通信中的从设备忙",
    0x750C: "Modbus 同步请求数据 > 映射总数据",
    0x750D: "Modbus 同步请求个数与映射不相等",
    0x750E: "Modbus 同步功能下单播报文节点地址错误",
    0x8311: "限扭矩保护",
    0x8610: "原点回归超时",
    0x8611: "位置超差",
    0x8613: "软件限位错误（暂停并锁轴）",
    0x8614: "限位开关错误（暂停并锁轴）",
    0x8615: "曲线规划计算错误",
    0x8616: "目标位置溢出故障",
    0x8617: "曲线规划参数过小",
    0xFF01: "电机参数识别故障",
    0xFF02: "参数保存故障",
    0xFF05: "STO_1 关闭失败故障",
    0xFF06: "STO_1 使能失败故障",
    0xFF07: "STO_2 关闭失败故障",
    0xFF08: "STO_2 使能失败故障",
    0xFF09: "STO 输入异常故障",
    0xFF0A: "STO 使能状态故障",
    0xFF0B: "非安全状态故障",
    0xFF0C: "编码器报警超限故障",
}


def decode_error_code_hex(err_hex: str) -> str:
    """
    根据 603Fh / 037Fh 读出的错误码（支持 16/32bit）给出解释。
    """
    if not err_hex:
        return "未读取到错误码（返回为空）。"

    try:
        full_val = int(err_hex, 16)
    except ValueError:
        return f"错误码格式异常：{err_hex!r}"

    # 低 16 位是真正的“错误码”
    low16 = full_val & 0xFFFF
    desc = ERROR_CODE_DESCRIPTIONS.get(low16, "未知错误码，请查阅手册第 8 章。")

    if len(err_hex) > 4:
        return f"错误码 0x{full_val:08X}（低 16 位 0x{low16:04X}）: {desc}"
    else:
        return f"错误码 0x{low16:04X}: {desc}"


def analyze_status(status_obj) -> str:
    """
    把 LMS_Status 对象转换为可读文字。
    """
    raw = status_obj.Raw()
    lines = [f"状态字 6041h = 0x{raw:04X}"]

    if status_obj.ReadyToSwitchOn():
        lines.append(" - bit0 Ready to switch on")
    if status_obj.SwitchedOn():
        lines.append(" - bit1 Switched on")
    if status_obj.OperationEnabled():
        lines.append(" - bit2 Operation enabled")
    if status_obj.Fault():
        lines.append(" - bit3 Fault = 1（有故障）")
    if status_obj.VoltageEnabled():
        lines.append(" - bit4 Voltage enabled（功放上电）")
    if status_obj.QuickStopActive():
        lines.append(" - bit5 Quick stop active")
    if status_obj.SwitchOnDisabled():
        lines.append(" - bit6 Switch on disabled")
    if status_obj.Warning():
        lines.append(" - bit7 Warning（有报警）")
    if status_obj.Remote():
        lines.append(" - bit9 Remote（远程控制）")
    if status_obj.TargetReached():
        lines.append(" - bit10 Target reached（到位）")
    if status_obj.InternalLimitActive():
        lines.append(" - bit11 Internal limit active（内部限位触发）")

    return "\n".join(lines)


# ==================== PySide6 GUI 部分 ====================

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QLabel,
    QMessageBox,
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import QTimer


class LMSTestWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.drv: LMS_Driver | None = None

        self._build_ui()

        # ★ 速度监控定时器
        self.speed_timer = QTimer(self)
        self.speed_timer.setInterval(100)  # 100ms 查询一次
        self.speed_timer.timeout.connect(self.poll_speed)

    # ---------- UI ----------
    def _build_ui(self):
        self.setWindowTitle("LMS 伺服测试工具")

        main_layout = QVBoxLayout(self)

        # 日志文本框
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(self.log_edit)

        # 连接按钮
        row_connect = QHBoxLayout()
        self.connect_button = QPushButton("Connect / Init")
        self.connect_button.clicked.connect(self.on_connect_clicked)
        row_connect.addWidget(self.connect_button)
        main_layout.addLayout(row_connect)

        # 移动：文本框 + 按钮
        row_move = QHBoxLayout()
        row_move.addWidget(QLabel("相对位移（用户单位/步）："))
        self.move_edit = QLineEdit("10000")
        row_move.addWidget(self.move_edit)
        self.move_button = QPushButton("Move")
        self.move_button.clicked.connect(self.on_move_clicked)
        row_move.addWidget(self.move_button)
        main_layout.addLayout(row_move)

        # 故障复位 & 查询
        row_ops = QHBoxLayout()
        self.fr_button = QPushButton("故障复位 (Fault Reset)")
        self.fr_button.clicked.connect(self.on_fr_clicked)
        row_ops.addWidget(self.fr_button)

        self.query_button = QPushButton("查询状态 / 报警")
        self.query_button.clicked.connect(self.on_query_clicked)
        row_ops.addWidget(self.query_button)

        main_layout.addLayout(row_ops)

        # 初始时除 connect 外都禁用
        self._set_connected(False)

    # ---------- 日志 ----------
    def log(self, msg: str):
        for line in str(msg).splitlines():
            ts = time.strftime("%H:%M:%S")
            self.log_edit.append(f"[{ts}] {line}")
        # 自动滚动到底部
        cursor = self.log_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_edit.setTextCursor(cursor)
        self.log_edit.ensureCursorVisible()

    def _set_connected(self, connected: bool):
        self.move_button.setEnabled(connected)
        self.fr_button.setEnabled(connected)
        self.query_button.setEnabled(connected)

    # ---------- 槽函数 ----------
    def on_connect_clicked(self):
        if self.drv is not None:
            self.log("已经连接，如需重新连接请重启程序。")
            return

        try:
            self.log("正在连接并初始化伺服...")
            drv = init_lms()
            reset(drv)
            self.drv = drv
            self._set_connected(True)
            self.log("连接 & 初始化完成。")
        except Exception as e:
            self.drv = None
            self._set_connected(False)
            self.log(f"连接失败：{e!r}")
            QMessageBox.critical(self, "连接失败", str(e))

    def on_move_clicked(self):
        if self.drv is None:
            self.log("尚未连接伺服，请先点击 Connect。")
            return

        text = self.move_edit.text().strip()
        try:
            distance = int(text)
        except ValueError:
            self.log("相对位移请输入整数。")
            return

        self.log(f"发送相对位移命令：{distance} 步/用户单位")
        try:
            runRelativePosition(self.drv, distance)
            self.start_speed_monitor()
        except Exception as e:
            self.log(f"移动失败：{e!r}")
            QMessageBox.warning(self, "移动失败", str(e))

    # ---------- 速度监控 ----------
    def start_speed_monitor(self):
        if self.drv is None:
            return
        if not self.speed_timer.isActive():
            self.log("开始监控电机转速（606Ch，单位 rpm）...")
            self.speed_timer.start()

    def stop_speed_monitor(self):
        if self.speed_timer.isActive():
            self.speed_timer.stop()
            self.log("停止速度监控。")

    def poll_speed(self):
        """
        定时器回调：每 100ms 读一次当前转速和状态字。
        当检测到目标已到达 / 电机不再使能 且 速度接近 0 时，自动停止监控。
        """
        if self.drv is None:
            self.speed_timer.stop()
            return

        try:
            # 606Ch（03D5h），单位 rpm
            velocity = self.drv.getActualVelocity()
            self.log(f"当前速度: {velocity} rpm")

            # 再看一下状态字，判断是否已经到位 / 停止
            st = self.drv.Status()
            if (st.TargetReached() or not st.OperationEnabled()) and abs(velocity) < 5:
                # 速度已经很小，认为停止了
                self.stop_speed_monitor()

        except Exception as e:
            self.log(f"读取速度失败: {e!r}")
            self.speed_timer.stop()


    def on_fr_clicked(self):
        if self.drv is None:
            self.log("尚未连接伺服，请先点击 Connect。")
            return

        self.log("执行 Fault Reset...")
        try:
            FR(self.drv)
            self.log("Fault Reset 命令已发送。")
        except Exception as e:
            self.log(f"Fault Reset 失败：{e!r}")
            QMessageBox.warning(self, "Fault Reset 失败", str(e))

    def on_query_clicked(self):
        if self.drv is None:
            self.log("尚未连接伺服，请先点击 Connect。")
            return

        try:
            status = self.drv.Status()
            self.log("===== 状态字查询 =====")
            self.log(analyze_status(status))

            if status.Fault() or status.Warning():
                self.log("检测到 Fault 或 Warning，读取错误码 603Fh (037Fh)...")
                # 读取 16bit 报警代码（1个寄存器）
                err_hex = self.drv.read(LMS.ErrorCode, 1)
                self.log(f"原始错误码寄存器值：{err_hex or '<空>'}")
                self.log(decode_error_code_hex(err_hex))
            else:
                self.log("当前状态字中 Fault(bit3) 与 Warning(bit7) 均为 0。")

            self.log("======================")
        except Exception as e:
            self.log(f"查询失败：{e!r}")
            QMessageBox.warning(self, "查询失败", str(e))


# ==================== 程序入口 ====================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LMSTestWindow()
    win.resize(700, 400)
    win.show()
    sys.exit(app.exec())
