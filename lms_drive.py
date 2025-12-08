# lms_c12.py
# 一体化低压伺服电机 LMS-C12..04 Modbus 驱动
# ModbusRTU 复用 dm2c.py 中的实现

import time
import copy

from dm2c import ModbusRTU  # 复用你现有的 ModbusRTU（以及 CRCint）


class LMS(object):
    """
    LMS 一体化低压伺服电机 Modbus 寄存器与常量定义
    手册：LMS-C12..04 一体化低压伺服电机 Modbus 通信（闭环）用户手册
    主要参考：
      - 第 3 章 设备控制（CiA402 状态机、控制字/状态字）
      - 第 5 章 运行模式（PP/PV/VM 等）
      - 第 10 章 对象字典（索引 -> Modbus 地址）
    """

    # ---------------- 通信 / 功能码 --------------
    ModbusFunctionCode_ReadRegs = b"\x03"
    ModbusFunctionCode_WriteReg = b"\x06"
    ModbusFunctionCode_WriteRegs = b"\x10"

    # ---------------- 模式选择 -------------------
    # 2002h:01h (00B1h) 控制模式选择（5.1 章节）
    # 0x00： CiA402 模式
    # 0x01： NiMotion 位置模式
    # 0x02： NiMotion 速度模式
    # 0x03： NiMotion 转矩模式
    # 0x04： NiMotion 开环模式
    ControlModeSelect = "00B1"  # 2002h:01h

    CONTROL_MODE_CIA402 = 0x00
    CONTROL_MODE_NIMOTION_POSITION = 0x01
    CONTROL_MODE_NIMOTION_VELOCITY = 0x02
    CONTROL_MODE_NIMOTION_TORQUE = 0x03
    CONTROL_MODE_NIMOTION_OPENLOOP = 0x04

    # 6060h (03C2h) 模式选择（5.1 章节）
    ModeOfOperation = "03C2"  # 6060h
    ModeOfOperationDisplay = "03C3"  # 6061h，只读显示当前模式

    # 6060h 的模式值
    MODE_UNDEFINED = 0x00
    MODE_PP = 0x01  # 轮廓位置模式
    MODE_VM = 0x02  # 速度模式（VM）
    MODE_PV = 0x03  # 轮廓速度模式
    MODE_PT = 0x04  # 轮廓转矩模式
    MODE_HM = 0x06  # 原点回归模式
    MODE_IP = 0x07  # 插补模式
    MODE_CSP = 0x08  # 循环同步位置
    MODE_CSV = 0x09  # 循环同步速度
    MODE_CST = 0x0A  # 循环同步转矩

    # -------- 数字输入 DI 功能/逻辑配置（对象 2003h）--------
    DI1_Function = "00D5"  # 2003h:03h  DI1 端子功能选择
    DI1_Logic = "00D6"  # 2003h:04h  DI1 端子逻辑选择
    DI2_Function = "00D7"  # 2003h:05h  DI2 端子功能选择
    DI2_Logic = "00D8"  # 2003h:06h  DI2 端子逻辑选择

    # ---------------- CiA402 控制字/状态字 ---------------
    # 6040h (0380h) 控制字
    ControlWord = "0380"
    # 6041h (0381h) 状态字
    StatusWord = "0381"

    # 一些常用 CiA402 控制字命令（参考 3.1 + 各模式示例）
    # 手册示例中多次给出：6040h = 0x06 -> 0x07 -> 0x0F 开机使能
    CW_SHUTDOWN = 0x0006          # Shutdown
    CW_SWITCH_ON = 0x0007         # Switched on
    CW_ENABLE_OPERATION = 0x000F  # Operation enabled
    CW_DISABLE_VOLTAGE = 0x0000   # 失能/掉电
    CW_QUICK_STOP = 0x0002        # Quick stop（bit1=1, bit2=0）
    CW_FAULT_RESET = 0x0080       # Fault reset（bit7）

    # 在 PP 模式下的额外位（5.3 章节表 5-7, 5-8）
    CW_BIT_NEW_SETPOINT = 1 << 4       # bit4 New set-point
    CW_BIT_CHANGE_IMMEDIATELY = 1 << 5 # bit5 Change set immediately
    CW_BIT_RELATIVE = 1 << 6           # bit6 (0: 绝对, 1: 相对)
    CW_BIT_HALT = 1 << 8               # bit8 Halt

    # ---------------- 位置相关对象 -----------------
    # 6062h (03C4h) 驱动器内部当前目标位置指令值（用户单位）
    InternalTargetPosition = "03C4"

    # 6064h (03C8h) 电机当前的用户绝对位置反馈（用户单位）
    ActualPosition = "03C8"

    # 607Ah (03E7h) 预设的目标位置（用户单位）
    TargetPosition = "03E7"

    # 607Dh (03EFh/03F1h) 目标位置限制（最小/最大）
    SoftwarePosLimitMin = "03EF"  # 607Dh:01h
    SoftwarePosLimitMax = "03F1"  # 607Dh:02h

    # 607Eh (03F3h) 极性（旋转方向）等，可根据需要再扩展
    Polarity = "03F3"  # 607Eh

    # 607Fh (03F4h) 最大轮廓速度（用户单位/s）
    MaxProfileVelocity = "03F4"

    # 6080h (03F6h) 电机最大转速（rpm）
    MaxMotorSpeed = "03F6"

    # ---------------- 速度 / 加减速 -----------------
    # 6081h (03F8h) 轮廓速度（PP 模式中当前段位移的匀速段速度）
    ProfileVelocity = "03F8"

    # 6082h (03FAh) 轮廓终点速度
    ProfileEndVelocity = "03FA"

    # 6083h (03FCh) 轮廓加速度
    ProfileAcceleration = "03FC"

    # 6084h (03FEh) 轮廓减速度
    ProfileDeceleration = "03FE"

    # 6085h (0400h) 快速停机减速度
    QuickStopDeceleration = "0400"

    # 6086h (0402h) 斜坡/曲线类型（0：梯形，3：S 型）
    ProfileJerkType = "0402"

    # 60C5h (043Bh) 最大加速度限值
    MaxAccelerationLimit = "043B"
    # 60C6h (043Dh) 最大减速度限值
    MaxDecelerationLimit = "043D"

    # ---------------- VM / PV / CSV 速度对象 ---------
    # -------- VM 速度模式相关对象（5.4.2） --------
    # 6042h / 6043h / 6044h
    VM_TargetSpeed = "0382"  # 6042h  VM模式目标速度
    VM_TargetSpeedEffective = "0383"  # 6043h  实际生效的目标速度（只读）
    VM_ActualSpeed = "0384"  # 6044h  当前实际速度（rpm）

    # 6046h:01/02   VM 模式速度限值
    VM_SpeedLimitMin = "0385"  # 6046h:01h 最低速度
    VM_SpeedLimitMax = "0387"  # 6046h:02h 最高速度

    # 6048h:01/02   VM 加速度参数
    VM_AccelDeltaSpeed = "0389"  # 6048h:01h  Δspeed (rpm * 0.1)
    VM_AccelDeltaTime = "038B"  # 6048h:02h  Δtime  (s   * 0.1)

    # 6049h:01/02   VM 减速度参数
    VM_DecelDeltaSpeed = "038C"  # 6049h:01h
    VM_DecelDeltaTime = "038E"  # 6049h:02h

    # 604Ah:01/02   VM 快速停机减速度
    VM_QSDeltaSpeed = "038F"  # 604Ah:01h
    VM_QSDeltaTime = "0391"  # 604Ah:02h

    # 604Ch:01/02   VM 速度单位缩放
    VM_SpeedScaleNum = "0394"  # 604Ch:01h 分子
    VM_SpeedScaleDen = "0396"  # 604Ch:02h 分母


    # 606Ch (03D5h) 当前实际速度反馈（rpm）
    ActualVelocity = "03D5"

    # 60FFh (0448h) 目标速度（用户单位/s）（PV/CSV 等）
    TargetVelocity = "0448"

    # ---------------- 到位 / 跟随误差 ----------------
    # 6065h (03CAh) 位置偏差过大阈值
    FollowingErrorWindow = "03CA"

    # 6066h (03CCh) 位置偏差过大超时
    FollowingErrorTimeout = "03CC"

    # 6067h (03CDh) 位置到达阈值
    PositionWindow = "03CD"

    # 6068h (03CFh) 位置到达时间窗口
    PositionWindowTime = "03CF"

    # ---------------- 原点回归相关（简略） -------------
    # 6099h:01h (0417h) 找开关速度（高速）
    HomingSwitchSpeed = "0417"

    # 6099h:02h (0419h) 找原点速度（低速）
    HomingZeroSpeed = "0419"

    # 609Ah (041Bh) 原点回归加速度
    HomingAcceleration = "041B"

    # （6098h 原点回归方式，对应 Modbus 地址 0415h，若需要可以再补）

    # ---------------- 错误 / 报警 --------------------
    # 603Fh (037Fh) 错误码
    ErrorCode = "037F"

    # ---------------- 方向语义（方便外部使用） --------
    Positive = True
    Negative = False

    Encoder = 10000
    VelocityRatio = Encoder / 60


class LMS_Status(object):
    """
    解析 6041h 状态字（CiA402 标准定义）
    """

    def __init__(self, driver):
        self.update(driver)

    def update(self, driver):
        status_hex = driver.read(LMS.StatusWord)
        if not status_hex:
            self.__Connected = False
            status = 0
        else:
            self.__Connected = True
            status = int(status_hex, 16)

        self.__raw = status

        # 参考手册表 3-4：6041h 各位含义
        self.__ReadyToSwitchOn = bool(status & (1 << 0))
        self.__SwitchedOn = bool(status & (1 << 1))
        self.__OperationEnabled = bool(status & (1 << 2))
        self.__Fault = bool(status & (1 << 3))
        self.__VoltageEnabled = bool(status & (1 << 4))
        self.__QuickStopActive = bool(status & (1 << 5))
        self.__SwitchOnDisabled = bool(status & (1 << 6))
        self.__Warning = bool(status & (1 << 7))
        self.__Remote = bool(status & (1 << 9))
        self.__TargetReached = bool(status & (1 << 10))
        self.__InternalLimitActive = bool(status & (1 << 11))
        # 12~13 为模式相关，视需要再解析

    # 一些访问接口
    def Raw(self):
        return self.__raw

    def Connected(self):
        return self.__Connected

    def ReadyToSwitchOn(self):
        return self.__ReadyToSwitchOn

    def SwitchedOn(self):
        return self.__SwitchedOn

    def OperationEnabled(self):
        return self.__OperationEnabled

    def Fault(self):
        return self.__Fault

    def VoltageEnabled(self):
        return self.__VoltageEnabled

    def QuickStopActive(self):
        return self.__QuickStopActive

    def SwitchOnDisabled(self):
        return self.__SwitchOnDisabled

    def Warning(self):
        return self.__Warning

    def Remote(self):
        return self.__Remote

    def TargetReached(self):
        return self.__TargetReached

    def InternalLimitActive(self):
        return self.__InternalLimitActive


class LMS_Driver(object):
    """
    LMS 伺服驱动封装，接口风格尽量和原 DM2C_Driver 类似：
      - connect / read / write 保持相同用法
      - 新增 CiA402 和 PP/PV 相关高层控制函数
    """

    def __init__(self, *args):
        self.__modbus = None
        self.__id = None

        self.regAddress = ""
        self.regNumber = ""
        self.regData = ""

        self.__status = None

        self.DI_FUNC_POS_LIMIT = 14
        self.DI_FUNC_NEG_LIMIT = 15

        if args:
            for arg in args:
                if isinstance(arg, bytes):
                    self.setIdx(arg)
                elif isinstance(arg, ModbusRTU):
                    self.setModbus(arg)

    # ---------------- 基本连接 & Modbus 封装 ----------------

    def connect(self, modbus, idx):
        self.setModbus(modbus)
        self.setIdx(idx)

    def setModbus(self, modbus):
        self.__modbus = modbus
        # 保持和 DM2C 一致，先关闭一次串口
        self.__modbus.closeCom()

    def setIdx(self, idx):
        self.__id = idx

    def comErr(self, msg=""):
        if msg:
            print("Modbus: com error:", msg)
        else:
            print("Modbus: com error")
        self.__modbus.closeCom()
        return 0

    def comDone(self):
        self.__modbus.closeCom()
        return 1

    def read(self, address, number=1):
        """
        读取寄存器（保持和 DM2C_Driver 接口一致）
        :param address: 4 位十六进制字符串，如 "0380"
        :param number: 寄存器数量（16bit 为单位）
        :return: 读取到的十六进制字符串（大写），长度 = number * 4
        """
        while self.__modbus.isBusy():
            pass
        print("R:", address, "x", number)
        self.__modbus.setBusy(True)
        self.regAddress = address
        self.regNumber = "%04X" % number
        if self.readRegs():
            print("R:", address, self.regData)
            self.__modbus.setBusy(False)
            return self.regData
        self.__modbus.setBusy(False)
        return ""

    def write(self, address, data):
        """
        写寄存器，data 为十六进制字符串：
          - 1 个 16bit：len(data) == 4
          - 32bit：len(data) == 8（会拆成两个连续寄存器）
          - 多个寄存器同理
        """
        number = int(len(data) / 4)
        while self.__modbus.isBusy():
            pass
        self.__modbus.setBusy(True)
        print("W:", address, data)
        self.regAddress = address
        self.regData = data
        if number == 1:
            self.writeReg()
            self.__modbus.setBusy(False)
        else:
            self.regNumber = "%04X" % number
            self.writeRegs()
            self.__modbus.setBusy(False)

    def readRegs(self):
        self.__modbus.address = self.__id
        self.__modbus.function = LMS.ModbusFunctionCode_ReadRegs
        self.__modbus.data = bytes.fromhex(self.regAddress + self.regNumber)
        if not self.__modbus.tx():
            return self.comErr("tx")
        if self.__modbus.rx(2) != self.__modbus.getMessage()[:2]:
            return self.comErr("addr+func mismatch")
        bytenumber = int(self.regNumber, 16) * 2
        if self.__modbus.rx(1) != bytes([bytenumber]):
            return self.comErr("byte count mismatch")
        self.__modbus.data = bytes([bytenumber]) + self.__modbus.rx(bytenumber)
        self.__modbus.calcCRC()
        if self.__modbus.rx(2) != self.__modbus.getCRC():
            return self.comErr("CRC error")
        self.regData = self.__modbus.data[1:].hex().upper()
        return self.comDone()

    def writeReg(self):
        self.__modbus.address = self.__id
        self.__modbus.function = LMS.ModbusFunctionCode_WriteReg
        self.__modbus.data = bytes.fromhex(self.regAddress + self.regData)
        if not self.__modbus.tx():
            return self.comErr("tx")
        if self.__modbus.rx(8) != self.__modbus.getMessage():
            return self.comErr("echo mismatch")
        return self.comDone()

    def writeRegs(self):
        self.__modbus.address = self.__id
        self.__modbus.function = LMS.ModbusFunctionCode_WriteRegs
        self.__modbus.data = (
            bytes.fromhex(self.regAddress + self.regNumber)
            + bytes([int(len(self.regData) / 2)])
            + bytes.fromhex(self.regData)
        )
        if not self.__modbus.tx():
            return self.comErr("tx")
        if self.__modbus.rx(6) != self.__modbus.getMessage()[:6]:
            return self.comErr("echo mismatch")
        self.__modbus.data = self.__modbus.data[:4]
        self.__modbus.calcCRC()
        if self.__modbus.rx(2) != self.__modbus.getCRC():
            return self.comErr("CRC error")
        return self.comDone()

    # ---------------- 状态查询 ----------------

    def Status(self):
        self.__status = LMS_Status(self)
        return copy.copy(self.__status)



    def isConnected(self):
        return self.Status().Connected()

    def isFault(self):
        return self.Status().Fault()

    def isOperationEnabled(self):
        return self.Status().OperationEnabled()

    def isRunning(self):
        return self.Status().TargetReached()

    def isTargetReached(self):
        return self.Status().TargetReached()

    # ---------------- 基本控制：模式 & 状态机 ----------------

    def setControlMode(self, mode):
        """
        设置 2002h:01h 控制模式（CiA402 / NiMotion）
        """
        self.write(LMS.ControlModeSelect, "%04X" % (mode & 0xFF))

    def setOperationMode(self, mode):
        """
        设置 6060h 模式选择（PP / PV / VM / HM 等）
        """
        self.write(LMS.ModeOfOperation, "%04X" % (mode & 0xFF))

    def enableOperation(self):
        """
        CiA402 标准的开机流程：
          6040h = 0x06 -> 0x07 -> 0x0F
        手册中多处示例也是这么写，执行完后电机进入使能状态。
        """
        self.write(LMS.ControlWord, "%04X" % LMS.CW_SHUTDOWN)
        self.write(LMS.ControlWord, "%04X" % LMS.CW_SWITCH_ON)
        self.write(LMS.ControlWord, "%04X" % LMS.CW_ENABLE_OPERATION)

    def shutdown(self):
        """
        切回 Shutdown 状态（保持 Ready to switch on）
        """
        self.write(LMS.ControlWord, "%04X" % LMS.CW_SHUTDOWN)

    def disableVoltage(self):
        """
        关闭功放（Disable voltage）
        """
        self.write(LMS.ControlWord, "%04X" % LMS.CW_DISABLE_VOLTAGE)

    def quickStop(self):
        """
        Quick stop（具体减速度由 605Ah/6084/6085 配置）
        """
        self.write(LMS.ControlWord, "%04X" % LMS.CW_QUICK_STOP)

    def faultReset(self):
        """
        故障复位：控制字 bit7 = 1
        一般做法：先写 0x80，再写 0x00，然后重新执行 enableOperation。
        """
        self.write(LMS.ControlWord, "%04X" % LMS.CW_FAULT_RESET)
        # 通常需要清零一次
        self.write(LMS.ControlWord, "%04X" % 0x0000)

    def init_limit_inputs(self,
                          di1_func=None,
                          di1_logic=0,
                          di2_func=None,
                          di2_logic=0):

        if di1_func is None:
            di1_func = self.DI_FUNC_POS_LIMIT
        if di2_func is None:
            di2_func = self.DI_FUNC_NEG_LIMIT

        # 这四个寄存器都是 uint16，所以用 4 位十六进制（1 个寄存器）
        # DI1 功能
        self.write(LMS.DI1_Function, "%04X" % (int(di1_func) & 0xFFFF))
        # DI1 逻辑（0 = 低电平有效）
        self.write(LMS.DI1_Logic, "%04X" % (int(di1_logic) & 0xFFFF))

        # DI2 功能
        self.write(LMS.DI2_Function, "%04X" % (int(di2_func) & 0xFFFF))
        # DI2 逻辑
        self.write(LMS.DI2_Logic, "%04X" % (int(di2_logic) & 0xFFFF))

    # ---------------- 位置 / 速度读取 ----------------

    @staticmethod
    def _to_signed32(raw_hex):
        val = int(raw_hex, 16)
        if val & 0x80000000:
            val -= 0x100000000
        return val

    def getActualPosition(self):
        """
        读取 6064h 当前用户绝对位置（用户单位）
        """
        raw = self.read(LMS.ActualPosition, 2)
        if not raw:
            return 0
        return self._to_signed32(raw)

    def getInternalTargetPosition(self):
        """
        读取 6062h 内部当前目标位置指令（用户单位）
        """
        raw = self.read(LMS.InternalTargetPosition, 2)
        if not raw:
            return 0
        return self._to_signed32(raw)

    def getActualVelocity(self):
        """
        读取 606Ch 当前实际速度反馈（rpm）
        """
        raw = self.read(LMS.ActualVelocity, 2)
        if not raw:
            return 0
        return self._to_signed32(raw)

    # ---------------- 轮廓位置模式（PP）控制 ----------------

    def initPPMode(self):
        """
        初始化为 CiA402 + PP 模式（不负责 enableOperation）
        """
        self.setControlMode(LMS.CONTROL_MODE_CIA402)
        self.setOperationMode(LMS.MODE_PP)

    def movePP(
        self,
        target_pos,
        velocity,
        accel,
        decel,
        relative=False,
        immediate=True,
        wait=False,
        poll_interval=0.03,
    ):
        """
        轮廓位置模式（PP）运动命令，参考 5.3 章节：
          - 写目标位置 607Ah
          - 设置 6081h（轮廓速度）、6083h（加速度）、6084h（减速度）
          - 通过 6040h bit4/bit5/bit6 触发运动

        :param target_pos: 目标位置（用户单位）
        :param velocity: 轮廓速度（用户单位/s）
        :param accel: 加速度（用户单位/s^2）
        :param decel: 减速度（用户单位/s^2）
        :param relative: True 表示相对位移（bit6=1），False 绝对位置
        :param immediate: True: 立刻更新（bit5=1），False: 非立刻更新（bit5=0）
        :param wait: True 则阻塞等待 TargetReached
        :param poll_interval: 轮询间隔（秒）
        """
        # 写目标位置 & 轮廓参数（int32）
        self.write(LMS.TargetPosition, "%08X" % (int(target_pos) & 0xFFFFFFFF))
        self.write(LMS.MaxProfileVelocity, "%08X" % (int(velocity) & 0xFFFFFFFF))
        self.write(LMS.ProfileVelocity, "%08X" % (int(velocity) & 0xFFFFFFFF))
        self.write(LMS.ProfileAcceleration, "%08X" % (int(accel) & 0xFFFFFFFF))
        self.write(LMS.ProfileDeceleration, "%08X" % (int(decel) & 0xFFFFFFFF))

        # 生成控制字
        # 先从“已使能”的基值 0x0F 开始
        cw = LMS.CW_ENABLE_OPERATION

        # 是否相对位置（bit6）
        if relative:
            cw |= LMS.CW_BIT_RELATIVE

        # 是否立即更新（bit5）
        if immediate:
            cw |= LMS.CW_BIT_CHANGE_IMMEDIATELY

        # 触发 new set-point：bit4 0 -> 1
        cw_no_new = cw & ~LMS.CW_BIT_NEW_SETPOINT
        cw_new = cw | LMS.CW_BIT_NEW_SETPOINT

        # 先写一个 bit4=0 的值，再写 bit4=1，形成上升沿
        self.write(LMS.ControlWord, "%04X" % cw_no_new)
        self.write(LMS.ControlWord, "%04X" % cw_new)

        # 一般可以再清一下 bit4（可选）
        self.write(LMS.ControlWord, "%04X" % cw_no_new)

        if wait:
            # 轮询状态字 bit10（Target reached）
            while True:
                status = self.Status()
                if status.TargetReached():
                    break
                time.sleep(poll_interval)

    # ---------------- 轮廓速度模式（PV）控制 ----------------

    def initPVMode(self):
        """
        初始化为 CiA402 + 轮廓速度模式（PV）
        """
        self.setControlMode(LMS.CONTROL_MODE_CIA402)
        self.setOperationMode(LMS.MODE_PV)

    def setPVParameters(self, velocity, accel, decel):
        """
        设置 PV 模式的目标速度与加减速参数（60FFh, 6083h, 6084h）
        注意：单位是“用户单位/s”，需要结合 608Fh/6091h 配置的位置单位换算。
        """
        self.write(LMS.TargetVelocity, "%08X" % (int(velocity) & 0xFFFFFFFF))
        self.write(LMS.ProfileAcceleration, "%08X" % (int(accel) & 0xFFFFFFFF))
        self.write(LMS.ProfileDeceleration, "%08X" % (int(decel) & 0xFFFFFFFF))

    def runPV(self, velocity, accel, decel):
        """
        以给定速度运行（PV 模式），需要：
          - 已调用 initPVMode()
          - 已 enableOperation()
        多次调用 runPV 只会更新速度与加减速，不会重新使能。
        """
        self.setPVParameters(velocity, accel, decel)

    def initVMMode(self):
        """
        切换到 CiA402 + VM 速度模式（6060h = 2），不自动使能。
        同时把 604Ch 的速度缩放设成 rpm（任意一个为 0 即为 rpm）。
        """
        self.setControlMode(LMS.CONTROL_MODE_CIA402)
        self.setOperationMode(LMS.MODE_VM)

        # 604Ch:01/02 用 16bit 写 0，表示单位就是 rpm
        self.write(LMS.VM_SpeedScaleNum, "%04X" % 0)
        self.write(LMS.VM_SpeedScaleDen, "%04X" % 0)

    def configVMAccelDecel(self,
                           accel_rpm, accel_time_s,
                           decel_rpm=None, decel_time_s=None,
                           qs_decel_rpm=None, qs_decel_time_s=None):
        """
        配置 VM 模式的加速度 / 减速度 / 快速停机减速度。

        手册例子：
          需要在 3.5s 内加速到 300rpm：
            6048h:01 = 3000
            6048h:02 = 35

        实际上就是：内部用 0.1rpm / 0.1s 的刻度存储：
          raw_speed = 300rpm * 10 = 3000
          raw_time  = 3.5s   * 10 = 35

        这里接口用“真实物理量”（rpm / s），我们内部乘 10 再写寄存器。
        """
        if decel_rpm is None:
            decel_rpm = accel_rpm
        if decel_time_s is None:
            decel_time_s = accel_time_s
        if qs_decel_rpm is None:
            qs_decel_rpm = decel_rpm
        if qs_decel_time_s is None:
            qs_decel_time_s = decel_time_s

        def to_raw_01(val):
            # 转 0.1 单位，并转为非负整数
            return max(0, int(round(val * 10.0)))

        # ---- 6048h:01 Δspeed（32bit）、6048h:02 Δtime（16bit） ----
        accel_speed_raw = to_raw_01(abs(accel_rpm))
        accel_time_raw = to_raw_01(accel_time_s)

        self.write(LMS.VM_AccelDeltaSpeed, "%08X" % (accel_speed_raw & 0xFFFFFFFF))
        self.write(LMS.VM_AccelDeltaTime, "%04X" % (accel_time_raw & 0xFFFF))

        # ---- 6049h:01 Δspeed（32bit）、6049h:02 Δtime（16bit） ----
        decel_speed_raw = to_raw_01(abs(decel_rpm))
        decel_time_raw = to_raw_01(decel_time_s)

        self.write(LMS.VM_DecelDeltaSpeed, "%08X" % (decel_speed_raw & 0xFFFFFFFF))
        self.write(LMS.VM_DecelDeltaTime, "%04X" % (decel_time_raw & 0xFFFF))

        # ---- 604Ah:01 Δspeed（32bit）、604Ah:02 Δtime（16bit） ----
        qs_speed_raw = to_raw_01(abs(qs_decel_rpm))
        qs_time_raw = to_raw_01(qs_decel_time_s)

        self.write(LMS.VM_QSDeltaSpeed, "%08X" % (qs_speed_raw & 0xFFFFFFFF))
        self.write(LMS.VM_QSDeltaTime, "%04X" % (qs_time_raw & 0xFFFF))

    def setVMSpeedRpm(self, speed_rpm):
        """
        写 VM 模式目标速度（6042h，单位 rpm，16bit）。
        """
        self.write(LMS.VM_TargetSpeed, "%04X" % (int(speed_rpm) & 0xFFFF))

    def setVMSpeedLimit(self, min_rpm, max_rpm):
        """
        设置 VM 模式的速度上下限（6046h:01/02，16bit）。
        """
        self.write(LMS.VM_SpeedLimitMin, "%04X" % (int(min_rpm) & 0xFFFF))
        self.write(LMS.VM_SpeedLimitMax, "%04X" % (int(max_rpm) & 0xFFFF))

    def enableOperationVM(self):
        """
        按手册 5.4.5 示例：
          6040h = 0x06 → 0x07 → 0x7F
        使能电机并启动 VM 速度规划器。
        """
        self.write(LMS.ControlWord, "%04X" % 0x0006)
        self.write(LMS.ControlWord, "%04X" % 0x0007)
        self.write(LMS.ControlWord, "%04X" % 0x007F)

    def runVM(self,
              speed_rpm,
              accel_time_s=1.0,
              decel_time_s=1.0,
              accel_rpm=None,
              decel_rpm=None,
              qs_decel_time_s=None,
              qs_decel_rpm=None):
        """
        一步到位：配置 VM 模式并按指定速度运行。

        :param speed_rpm: 目标速度（rpm，可为负，代表反转）
        :param accel_time_s: 从 0 加到 |speed_rpm| 的时间（秒）
        :param decel_time_s: 从 |speed_rpm| 减到 0 的时间（秒）
        """
        # 先切 VM 模式
        self.initVMMode()

        # 配加减速度
        self.configVMAccelDecel(
            accel_rpm=accel_rpm or abs(speed_rpm),
            accel_time_s=accel_time_s,
            decel_rpm=decel_rpm or abs(speed_rpm),
            decel_time_s=decel_time_s,
            qs_decel_rpm=qs_decel_rpm or abs(speed_rpm),
            qs_decel_time_s=qs_decel_time_s or decel_time_s,
        )

        # 设速度限幅（可以不调用，这里简单给个上限 = 2 倍目标）
        self.setVMSpeedLimit(
            min_rpm=0,
            max_rpm=max(1, abs(speed_rpm) * 2),
        )

        # 写目标速度（6042h）
        self.setVMSpeedRpm(speed_rpm)

        # 有故障先复位
        if self.isFault():
            self.faultReset()

        # 6040h 0x06 → 0x07 → 0x7F
        self.enableOperationVM()


    # ---------------- 简单原点回归（HM）占位接口 ----------------

    def initHMMode(self):
        """
        初始化为 CiA402 + 原点回归模式（HM）
        注：具体 6098h 原点方式、6099h 速度、609Ah 加速度的配置
            可以根据实际机械需求扩展，这里只提供模式切换。
        """
        self.setControlMode(LMS.CONTROL_MODE_CIA402)
        self.setOperationMode(LMS.MODE_HM)

    # 这里可以再根据 5.7 章节补充 Home 的完整流程：
    # - 写 6098h Homing method
    # - 写 6099h 高/低速，609Ah 加速度
    # - 通过 6040h bit4/bit8 启动/停止回零
    # 暂时留给你根据现场需求继续扩展。

    # ---------------- 错误读取 ----------------

    def getErrorCode(self):
        """
        读取 603Fh 错误码
        """
        return self.read(LMS.ErrorCode)


if __name__ == "__main__":
#     示例（伪代码）：
    import serial
    ser = serial.Serial("COM7", 115200, timeout=0.1)
    modbus = ModbusRTU(ser)
    drv = LMS_Driver(modbus, b"\x02")  # 驱动器地址 1





    # 6091h:01h  传动比 分子 (040Eh)
    num_hex = drv.read("040E", 2)  # 读 2 个寄存器（uint32）
    num_val = int(num_hex, 16)  # 转成 Python 整数（无符号）

    # 6091h:02h  传动比 分母 (0410h)
    den_hex = drv.read("0410", 2)
    den_val = int(den_hex, 16)

    delta = drv.read("0406", 2)  # 读 2 个寄存器（uint32）
    delta_val = int(delta, 16)  # 转成 Python 整数（无符号）
    print(delta_val)

    # 6091h:02h  传动比 分母 (0410h)
    motor = drv.read("0408", 2)
    motor_val = int(motor, 16)

    print("传动比分子:", num_val)
    print("传动比分母:", den_val)
    print("编码器增量:", delta_val)
    print("电机转数：", motor_val)

    a = drv.getActualPosition()
    print("position:", a)

    #
    # accel = int(amax * 0.8)
    # decel = int(dmax * 0.8)
    #
    # drv.write("03FC", "%08X" % accel)  # 6083h 轮廓加速度
    # drv.write("03FE", "%08X" % decel)  # 6084h 轮廓减速度
    # drv.write("0400", "%08X" % decel)  # 6085h Quick stop 减速度，顺便一起拉高
    #
    # drv.initPPMode()
    # drv.enableOperation()
    # i = 0
    # while i < 10:
    #     drv.movePP(
    #         target_pos=1000000,
    #         velocity=5000000,  # 你算好的对应 3000rpm 的那个
    #         accel=amax,
    #         decel=dmax,
    #         relative=True,
    #         wait=True,
    #     )
    #     i += 1
    #
    #
    #
    #
    # print("当前位置：", drv.getActualPosition())
    #
    # pass

# if __name__ == "__main__":
#     import serial
#     import time
#     from dm2c import ModbusRTU
#     from lms_drive import LMS_Driver, LMS
#
#     # 1. 打开串口 & 创建驱动对象
#     ser = serial.Serial("COM7", 115200, timeout=0.1)
#     modbus = ModbusRTU(ser)
#     drv = LMS_Driver(modbus, b"\x01")  # 驱动器地址 1
#
#     # 2. 读最大加减速度限制（60C5h / 60C6h）
#     amax_hex = drv.read("043B", 2)  # MaxAccelerationLimit
#     dmax_hex = drv.read("043D", 2)  # MaxDecelerationLimit
#
#     amax = int(amax_hex, 16)
#     dmax = int(dmax_hex, 16)
#     print("最大加速度限制:", amax)
#     print("最大减速度限制:", dmax)
#
#     accel = int(amax)
#     decel = int(dmax)
#
#     # 3. 切到 CiA402 + 轮廓速度模式（PV），并使能
#     drv.initPVMode()          # 设置 2002h/6060h
#     if drv.isFault():         # 有故障先复位
#         drv.faultReset()
#     drv.enableOperation()     # 6040h: 06 -> 07 -> 0F
#
#     def run_with_speed_monitor(set_velocity, duration=3.0, interval=0.1):
#         """
#         以 set_velocity 运行 duration 秒，
#         期间每隔 interval 秒读取一次 606Ch（实际速度，rpm）并打印。
#         """
#         print(f"\n设置目标速度: {set_velocity} (用户单位/s)，持续 {duration} 秒")
#         drv.runPV(velocity=set_velocity, accel=accel, decel=decel)
#
#         t0 = time.time()
#         while True:
#             now = time.time()
#             if now - t0 > duration:
#                 break
#
#             # 读取 606Ch (03D5h)，单位：rpm
#             actual_rpm = drv.getActualVelocity()
#             print(f"t={now - t0:5.2f}s, 实际速度: {actual_rpm} rpm")
#
#             time.sleep(interval)
#
#     try:
#         # ★ 根据你的单位计算好对应 3000rpm 的 set_velocity 值
#         # 这里先用你之前的例子：例如 3000rpm 对应 500000
#         v1 = 2500*  LMS.VelocityRatio    # 正向
#         v2 = 2750 * LMS.VelocityRatio   # 反向
#
#         # 正转监控
#         run_with_speed_monitor(v1, duration=3.0, interval=0.1)
#
#         # 反转监控
#         run_with_speed_monitor(v2, duration=3.0, interval=0.1)
#
#     finally:
#         # 5. 快速停机
#         print("\nQuick stop 停止电机")
#         drv.quickStop()
#         time.sleep(1)
#
#         print("停止后实际速度：", drv.getActualVelocity(), "rpm")
#         print("当前位置：", drv.getActualPosition())
#
#     status = drv.Status()


# if __name__ == "__main__":
#     import serial
#     import time
#     from dm2c import ModbusRTU
#
#     ser = serial.Serial("COM7", 115200, timeout=0.1)
#     modbus = ModbusRTU(ser)
#     drv = LMS_Driver(modbus, b"\x01")  # 改成你的从站地址
#
#     # 目标：1000rpm，加速 1s，减速 1s
#     drv.runVM(
#         speed_rpm=3000,
#         accel_time_s=0.5,
#         decel_time_s=0.5,
#     )
#
#     t0 = time.time()
#     while time.time() - t0 < 5.0:
#         v = drv.getActualVelocity()  # 606Ch（03D5h），单位 rpm
#         print(f"t={time.time() - t0:4.2f}s, 实际速度 = {v} rpm")
#         time.sleep(0.1)
#
#     drv.runVM(
#         speed_rpm=2000,
#         accel_time_s=0.5,
#         decel_time_s=0.5,
#     )
#
#     t0 = time.time()
#     while time.time() - t0 < 5.0:
#         v = drv.getActualVelocity()  # 606Ch（03D5h），单位 rpm
#         print(f"t={time.time() - t0:4.2f}s, 实际速度 = {v} rpm")
#         time.sleep(0.1)
#     print("Quick stop 停止电机")
#     drv.quickStop()
#     time.sleep(0.5)
#     print("停止后速度:", drv.getActualVelocity(), "rpm")
