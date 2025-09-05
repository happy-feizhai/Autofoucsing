import serial
import copy
import time

from serial.tools import list_ports

from dm2c import DM2C, DM2C_Driver, ModbusRTU


class Device_Ports(object):
    def SerialPort(self):
        return [port.name for port in list_ports.comports()]


class Stage_LinearMovement(DM2C_Driver):
    # def __init__(self, *args):
    #     super().__init__(args)

    def setAbsolutePositionPath(self, idx = 15, speed = 800, acceleration = 80, deceleration = 80):
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

    def setRelativePositionPath(self, idx = 14, speed = 800, acceleration = 80, deceleration = 80):
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

    def setSpeedPath(self, idx = 13, acceleration = 80, deceleration = 80):
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

    def goAbsolutePosition(self, position):
        self.app.setPosition(position)
        self.write(DM2C.PathPositionH[self.app.Idx()], "%08X" % self.app.Position())
        self.goPath(self.app)

    def goRelativePosition(self, position):
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


class Stage_RotaryMovement(DM2C_Driver):
    # def __init__(self, modbus, id):
    #     super().__init__(modbus, id)

    def reset(self):
        super().reset()
        self.write(DM2C.DI4, DM2C.DI_Origin_NormallyClose)


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
    modbus = ModbusRTU(serial.Serial(port = serialport, baudrate = 115200, timeout = 2, write_timeout = 2))

    xaxis = Stage_LinearMovement(DM2C.Driver_01)
    xaxis.setModbus(modbus)
    xaxis.reset()
    xaxis.setRelativePositionPath(speed = 800, acceleration = 80, deceleration = 80)
    xaxis.setAbsolutePositionPath()
    xaxis.setSpeedPath()
    xaxis.setJogSpeed(Device_Stage.X_LowSpeed, Device_Stage.X_HighSpeed)

    zaxis = Stage_LinearMovement(DM2C.Driver_03)
    zaxis.setModbus(modbus)
    zaxis.reset()
    zaxis.setRelativePositionPath()
    zaxis.setAbsolutePositionPath()
    zaxis.setSpeedPath()
    zaxis.setJogSpeed(Device_Stage.X_LowSpeed, Device_Stage.X_HighSpeed)

    xaxis.goRelativePosition(50000)

