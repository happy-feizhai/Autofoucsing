import serial
import copy
import time

from serial.tools import list_ports

from dm2c import DM2C, DM2C_Driver, ModbusRTU
from lms_drive import LMS_Driver, LMS


class Device_Ports(object):
    def SerialPort(self):
        return [port.name for port in list_ports.comports()]


# ======================= 原 DM2C 直线轴实现 =======================

class Stage_LinearMovement_DM2C(DM2C_Driver):
    """
    原来的 Stage_LinearMovement，只给 X / Z 这类 DM2C 步进电机使用。
    接口保持不变。
    """

    # def __init__(self, *args):
    #     super().__init__(args)

    def setAbsolutePositionPath(self, idx=15, speed=800, acceleration=80, deceleration=80):
        path = self.Path(idx)
        path.setType(DM2C.GoPosition)
        path.setInsert(False)
        path.setOverlap(False)
        path.setRelativePosition(False)
        path.setSpeed(speed)
        path.setAcceleration(acceleration)
        path.setDeceleration(deceleration)
        path.setJump(False)
        self.app = path
        self.addPath(path)

    def setRelativePositionPath(self, idx=14, speed=800, acceleration=80, deceleration=80):
        path = self.Path(idx)
        path.setType(DM2C.GoPosition)
        path.setInsert(False)
        path.setOverlap(False)
        path.setRelativePosition(True)
        path.setSpeed(speed)
        path.setAcceleration(acceleration)
        path.setDeceleration(deceleration)
        path.setJump(False)
        self.rpp = path
        self.addPath(path)

    def setSpeedPath(self, idx=13, acceleration=80, deceleration=80):
        path = self.Path(idx)
        path.setType(DM2C.GoSpeed)
        path.setInsert(False)
        path.setOverlap(False)
        path.setAcceleration(acceleration)
        path.setDeceleration(deceleration)
        path.setJump(False)
        self.sp = path
        self.addPath(path)

    def reset(self):
        super().reset()
        self.write(DM2C.DI2, DM2C.DI_LimitNegative_NormallyOpen)
        self.write(DM2C.DI3, DM2C.DI_LimitPositive_NormallyOpen)
        self.setZero()

    def goAbsolutePosition(self, position, wait):
        self.app.setPosition(position)
        self.write(DM2C.PathPositionH[self.app.Idx()], "%08X" % self.app.Position())
        self.goPath(self.app)

    def goRelativePosition(self, position, wait):
        self.rpp.setPosition(position)
        self.write(DM2C.PathPositionH[self.rpp.Idx()], "%08X" % self.rpp.Position())
        self.goPath(self.rpp)

    def goSpeed(self, speed):
        self.sp.setSpeed(speed)
        self.write(DM2C.PathSpeed[self.sp.Idx()], "%04X" % self.sp.Speed())
        self.goPath(self.sp)

    def setJogSpeed(self, lowspeed, highspeed):
        self.jogPositiveL = self.JogSetting()
        self.jogNegativeL = copy.copy(self.jogPositiveL)

        self.jogPositiveL.setDirection(DM2C.Positive)
        self.jogPositiveH = copy.copy(self.jogPositiveL)
        self.jogNegativeL.setDirection(DM2C.Negative)
        self.jogNegativeH = copy.copy(self.jogNegativeL)
        self.jogPositiveL.setSpeed(lowspeed)
        self.jogNegativeL.setSpeed(lowspeed)
        self.jogPositiveH.setSpeed(highspeed)
        self.jogNegativeH.setSpeed(highspeed)


# ======================= LMS 伺服轴适配器 =======================

class LMS_LinearMovement(object):
    """
    把 LMS_Driver 适配成和 Stage_LinearMovement 类似的接口，
    这样 MotorController 不用改，仍然调用：
        reset / setRelativePositionPath / setAbsolutePositionPath /
        setSpeedPath / setJogSpeed / goAbsolutePosition / goRelativePosition /
        goSpeed / Position / stop
    """

    def __init__(self, idx):
        # idx 就是旧代码里传进来的 DM2C.Driver_02（本质上是一个从站地址 bytes）
        self._idx = idx
        self._drv = LMS_Driver(idx)

        # 保存路径/速度参数（单位直接沿用你 AXIS_CONFIG 里的数值）
        self.abs_speed = 500000
        self.abs_accel = 4294967295
        self.abs_decel = 4294967295

        self.rel_speed = 500000
        self.rel_accel = 4294967295
        self.rel_decel = 4294967295

        self.speed_accel = 4294967295
        self.speed_decel = 4294967295

        self.jog_low = 100
        self.jog_high = 500

    def setModbus(self, modbus):
        self._drv.setModbus(modbus)

    def reset(self):
        """
        伺服的基本初始化：
          - 读取最大加减速度限制
          - 把轮廓加/减速度和 QuickStop 减速度拉到这个上限
          - 设置 CiA402 + PP 模式
          - 使能电机
        """
        # 读最大加/减速度限制
        amax_hex = self._drv.read(LMS.MaxAccelerationLimit, 2)
        dmax_hex = self._drv.read(LMS.MaxDecelerationLimit, 2)
        try:
            amax = int(amax_hex, 16) if amax_hex else 0
            dmax = int(dmax_hex, 16) if dmax_hex else 0
        except ValueError:
            amax = dmax = 0

        if amax and dmax:
            # 直接写入 6083h / 6084h / 6085h
            self._drv.write(LMS.ProfileAcceleration, "%08X" % amax)
            self._drv.write(LMS.ProfileDeceleration, "%08X" % dmax)
            self._drv.write(LMS.QuickStopDeceleration, "%08X" % dmax)

        #设置限位开关
        self._drv.init_limit_inputs()

        self._drv.write(LMS.ProfileJerkType, "0000")

        # 模式切换 & 使能
        self._drv.initPPMode()
        # 确保没有故障
        if self._drv.isFault():
            self._drv.faultReset()
        self._drv.enableOperation()

    # ---- 这三个只是记录参数，真正用在 goXXX 里 ----

    def setRelativePositionPath(self, idx=14, speed=800, acceleration=80, deceleration=80):
        self.rel_speed = speed
        self.rel_accel = acceleration
        self.rel_decel = deceleration

    def setAbsolutePositionPath(self, idx=15, speed=800, acceleration=80, deceleration=80):
        self.abs_speed = speed
        self.abs_accel = acceleration
        self.abs_decel = deceleration

    def setSpeedPath(self, idx=13, acceleration=80, deceleration=80):
        self.speed_accel = acceleration
        self.speed_decel = deceleration

    def setJogSpeed(self, lowspeed, highspeed):
        # 目前没有用到 Jog，只是把值存起来以防以后需要
        self.jog_low = lowspeed
        self.jog_high = highspeed

    # ---- 实际运动接口 ----

    def goAbsolutePosition(self, position, wait = False):
        """
        绝对位置移动 -> PP 模式，relative=False
        position 单位：直接沿用你原来代码的“steps”/用户单位，
        对应驱动器 607Ah 的用户单位。
        """
        self._drv.movePP(
            target_pos=position,
            velocity=self.abs_speed,
            accel=self.abs_accel,
            decel=self.abs_decel,
            relative=False,
            wait=wait,
        )

    def goRelativePosition(self, position, wait = False):
        """
        相对位置移动 -> PP 模式，relative=True
        """
        self._drv.movePP(
            target_pos=position,
            velocity=self.rel_speed,
            accel=self.rel_accel,
            decel=self.rel_decel,
            relative=True,
            wait=wait,
        )

    def goSpeed(self, speed):
        """
        给 MotorController 用的 go_speed() 适配：
          - speed=0 时执行快速停机
          - 非 0 时进入 PV 模式（轮廓速度），按给定速度运行
        目前主程序并没有对 Y 轴调用 go_speed，所以这部分更多是兜底实现。
        """
        if speed == 0:
            self.stop()
            return

        # 切到 PV 模式并运行
        self._drv.initPVMode()
        if self._drv.isFault():
            self._drv.faultReset()
        self._drv.enableOperation()
        self._drv.runPV(
            velocity=int(speed),
            accel=self.speed_accel,
            decel=self.speed_decel,
        )

    def Position(self):
        """
        对应 DM2C_Driver.Position()，返回当前用户位置（6064h）
        """
        return self._drv.getActualPosition()

    def isRunning(self):
        return not self._drv.isRunning()

    def stop(self):
        """
        快速停机
        """
        self._drv.quickStop()


# ======================= 统一对外的 Stage_LinearMovement =======================

class Stage_LinearMovement(object):
    """
    统一封装：
      - X / Z 轴：内部使用 Stage_LinearMovement_DM2C（原步进电机逻辑）
      - Y 轴：内部使用 LMS_LinearMovement（新的伺服电机）
    对外公开的接口与原 Stage_LinearMovement 完全一致，
    所以 temp.py 里的 MotorController 不需要做任何修改。
    """

    def __init__(self, driver):
        self._driver = driver

        if driver == DM2C.Driver_02:
            # Y 轴 -> 伺服
            self._impl = LMS_LinearMovement(driver)
        else:
            # 其他轴 -> 原 DM2C 步进
            self._impl = Stage_LinearMovement_DM2C(driver)

    # --- 把常用接口全部转发给内部实现 ---

    def setModbus(self, modbus):
        self._impl.setModbus(modbus)

    def reset(self):
        self._impl.reset()

    def setRelativePositionPath(self, *args, **kwargs):
        return self._impl.setRelativePositionPath(*args, **kwargs)

    def setAbsolutePositionPath(self, *args, **kwargs):
        return self._impl.setAbsolutePositionPath(*args, **kwargs)

    def setSpeedPath(self, *args, **kwargs):
        return self._impl.setSpeedPath(*args, **kwargs)

    def setJogSpeed(self, *args, **kwargs):
        return self._impl.setJogSpeed(*args, **kwargs)

    def goAbsolutePosition(self, *args, **kwargs):
        return self._impl.goAbsolutePosition(*args, **kwargs)

    def goRelativePosition(self, *args, **kwargs):
        return self._impl.goRelativePosition(*args, **kwargs)

    def goSpeed(self, *args, **kwargs):
        return self._impl.goSpeed(*args, **kwargs)

    def Position(self, *args, **kwargs):
        return self._impl.Position(*args, **kwargs)

    def stop(self, *args, **kwargs):
        return self._impl.stop(*args, **kwargs)
    def isRunning(self):
        return self._impl.isRunning()


# ======================= 旋转轴保持不变 =======================

class Stage_RotaryMovement(DM2C_Driver):
    # def __init__(self, modbus, id):
    #     super().__init__(modbus, id)

    def reset(self):
        super().reset()
        self.write(DM2C.DI4, DM2C.DI_Origin_NormallyClose)


# ======================= 简单自测（可选） =======================

if __name__ == '__main__':
    class Device_Stage(object):
        X = DM2C.Driver_01
        Y = DM2C.Driver_02
        Z = DM2C.Driver_03
        R = DM2C.Driver_04
        X_HighSpeed = 500
        X_LowSpeed = 100
        Y_HighSpeed = 500
        Y_LowSpeed = 100
        Z_HighSpeed = 500
        Z_LowSpeed = 100
        R_Speed = 200
        R_Acceleration = 800
        R_GoZeroHighSpeed = 100
        R_GoZeroLowSpeed = 10
        R_Pulse = 36000
        # gear ration is 10, so MeridianRatio = R_Pulse / 360 * 10
        MeridianRatio = 1000

    serialport = "COM7"
    modbus = ModbusRTU(serial.Serial(port=serialport, baudrate=115200, timeout=2, write_timeout=2))

    # X 轴（步进电机）
    xaxis = Stage_LinearMovement(Device_Stage.X)
    xaxis.setModbus(modbus)
    xaxis.reset()
    xaxis.setRelativePositionPath(speed=800, acceleration=80, deceleration=80)
    xaxis.setAbsolutePositionPath()
    xaxis.setSpeedPath()
    xaxis.setJogSpeed(Device_Stage.X_LowSpeed, Device_Stage.X_HighSpeed)

    # Z 轴（步进电机）
    # zaxis = Stage_LinearMovement(Device_Stage.Z)
    # zaxis.setModbus(modbus)
    # zaxis.reset()
    # zaxis.setRelativePositionPath()
    # zaxis.setAbsolutePositionPath()
    # zaxis.setSpeedPath()
    # zaxis.setJogSpeed(Device_Stage.Z_LowSpeed, Device_Stage.Z_HighSpeed)

    # Y 轴（伺服电机），如果已经接在 Driver_02 的地址上，也可以在这里简单测一下：
    # yaxis = Stage_LinearMovement(Device_Stage.Y)
    # yaxis.setModbus(modbus)
    # yaxis.reset()
    # yaxis.setRelativePositionPath(speed=1100, acceleration=50, deceleration=50)
    # yaxis.goRelativePosition(10000)

    # 简单运动测试：X 轴相对移动
    xaxis.goRelativePosition(-50000, wait=False)
