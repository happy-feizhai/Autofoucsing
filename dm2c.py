import copy
import time
from math import ceil


class CRCint(object):
    def __init__(self, Width, Poly, Init, RefIn, RefOut):
        self.Width = Width
        self.Poly = Poly
        self.__divisor = Poly + (1 << Width)
        self.Init = Init
        self.RefIn = RefIn
        self.RefOut = RefOut

    def calcInt(self, data):
        if self.RefIn:
            data = CRCint.byteturnaround(data)
        if self.Init:
            crc = CRCint.m2d((data << self.Width) ^ (self.Init << ceil(CRCint.bits(data) / 8) * 8), self.__divisor)
        else:
            crc = CRCint.m2d(data << self.Width, self.__divisor)
        if self.RefOut:
            crc = CRCint.widthturnaround(crc, self.Width)
        return crc

    def calcHexStr(self, data):
        # return '%x'%self.calcInt(int((data),16))
        if self.Width == 8:
            return '%02x' % self.calcInt(int(data, 16))
        elif self.Width == 16:
            return '%04x' % self.calcInt(int(data, 16))
        elif self.Width == 32:
            return '%08x' % self.calcInt(int(data, 16))

    def calcBytes(self, data):
        return bytes.fromhex(self.calcHexStr(data.hex()))

    @staticmethod
    def bits(n):
        i = 0
        while n > 1:
            n >>= 1
            i += 1
        return i + n

    # modulo 2 division
    @staticmethod
    def m2d(n, d):
        dbits = CRCint.bits(d)
        p = CRCint.bits(n) - dbits
        while p >= 0:
            nhead = n >> p
            n -= nhead << p
            n += (nhead ^ d) << p
            p = CRCint.bits(n) - dbits
        return n

    @staticmethod
    def turnaround(n):
        b = CRCint.bits(n)
        newn = 0
        while b > 0:
            b -= 1
            newn += (n & 1) << b
            n >>= 1
        return newn

    @staticmethod
    def widthturnaround(n, width):
        return CRCint.turnaround(n) << (width - CRCint.bits(n))

    @staticmethod
    def byteturnaround(n):
        if n < 0x100:
            return CRCint.widthturnaround(n, 8)
        else:
            return (CRCint.byteturnaround(n >> 8) << 8) + CRCint.widthturnaround(n & 0xff, 8)


class ModbusRTU(object):
    def __init__(self, com):
        self.__com = com
        self.closeCom()
        self.address = b''
        self.function = b''
        self.data = b''
        self.__busy = False
        self.__crc16 = CRCint(16, 0x8005, 0xffff, True, True)

    def calcCRC(self):
        tempbytes = self.__crc16.calcBytes(self.address + self.function + self.data)
        self.__crc = bytes([tempbytes[1], tempbytes[0]])

    def getCRC(self):
        return self.__crc

    def calcMessage(self):
        self.calcCRC()
        self.__message = self.address + self.function + self.data + self.__crc

    def getMessage(self):
        return self.__message

    def tx(self):
        self.openCom()
        self.calcMessage()
        try:
            self.__com.write(self.__message)
        except Exception as e:
            return False
        return True

    def rx(self, len):
        try:
            reply = self.__com.read(len)
        except Exception as e:
            print(e)
            return False
        return reply

    def closeCom(self):
        self.__com.close()

    def openCom(self):
        self.__com.close()
        self.__com.open()

    def setBusy(self, isbusy):
        self.__busy = isbusy

    def isBusy(self):
        return self.__busy


class DM2C(object):
    RunningDirection = "0007"
    RunningDirection_Positive = "0000"
    RunningDirection_Negative = "0001"

    Pulse = "0001"

    ControlWord = "1801"
    ControlWord_InitParameter = "2222"
    ControlWord_FactoryParameter = "2233"
    ControlWord_JogPositive = "4001"
    ControlWord_JogNegative = "4002"

    DI1 = "0145"
    DI2 = "0147"
    DI3 = "0149"
    DI4 = "014B"
    DI5 = "014D"
    DI6 = "014F"
    DI7 = "0151"
    DI_LimitPositive_NormallyOpen = "0025"
    DI_LimitNegative_NormallyOpen = "0026"
    DI_Origin_NormallyClose = "00A7"

    Status = "1003"
    Error = "2203"

    Trigger = "6002"
    Trigger_Path = ["%04X" % i for i in range(0x0010, 0x0020)]
    Trigger_PathRunning = ["%04X" % i for i in range(0x0100, 0x0110)]
    Trigger_PathDone = ["%04X" % i for i in range(0x0000, 0x0010)]
    Trigger_GoZero = "0020"
    Trigger_SetZero = "0021"
    Trigger_Stop = "0040"

    JogSpeed = "01E1"
    JogInterval = "01E3"
    JogCycle = "01E5"
    JogAcceleration = "01E7"

    TargetPositionH = "602A"
    TargetPositionL = "602B"
    CurrentPositionH = "602C"
    CurrentPositionL = "602D"

    GoZeroMode = "600A"
    GoZeroHighSpeed = "600F"
    GoZeroLowSpeed = "6010"
    GoZeroAcceleration = "6011"
    GoZeroDeceleration = "6012"
    GoZeroOvertravel = "6013"

    Path = ["%04X" % (0x6200 + 8 * i) for i in range(16)]
    PathPositionH = ["%04X" % (0x6201 + 8 * i) for i in range(16)]
    PathPositionL = ["%04X" % (0x6202 + 8 * i) for i in range(16)]
    PathSpeed = ["%04X" % (0x6203 + 8 * i) for i in range(16)]
    PathAcceleration = ["%04X" % (0x6204 + 8 * i) for i in range(16)]
    PathDeceleration = ["%04X" % (0x6205 + 8 * i) for i in range(16)]
    PathStopTime = ["%04X" % (0x6206 + 8 * i) for i in range(16)]
    PathMapping = ["%04X" % (0x6207 + 8 * i) for i in range(16)]

    NoMotion = 0
    GoPosition = 1
    GoSpeed = 2
    GoZero = 3

    ModbusFunctionCode_ReadRegs = b'\x03'
    ModbusFunctionCode_WriteReg = b'\x06'
    ModbusFunctionCode_WriteRegs = b'\x10'

    Positive = True
    Negative = False

    Origin = True
    Limit = False

    Driver_01 = b"\x01"
    Driver_02 = b"\x02"
    Driver_03 = b"\x03"
    Driver_04 = b"\x04"
    Driver_05 = b"\x05"
    Driver_06 = b"\x06"


class DM2C_Path(object):
    def __init__(self, driver, idx):
        self.setIdx(idx)
        self.update(driver)

    def update(self, driver):
        pathdata = driver.read(DM2C.Path[self.__Idx], number = 8)
        path = int(pathdata[0:4])
        self.setType(path & 0x000F)
        if path & 0x0010:
            self.setInsert(True)
        else:
            self.setInsert(False)

        if path & 0x0020:
            self.setOverlap(True)
        else:
            self.setOverlap(False)

        if path & 0x0040:
            self.setRelativePosition(True)
        else:
            self.setRelativePosition(False)

        self.setNextPath((path & 0x3F00) >> 8)
        if path & 0x4000:
            self.setJump(True)
        else:
            self.setJump(False)

        self.setPosition(int(pathdata[4:12]))
        self.setSpeed(int(pathdata[12:16]))
        self.setAcceleration(int(pathdata[16:20]))
        self.setDeceleration(int(pathdata[20:24]))
        self.setStopTime(int(pathdata[24:28]))
        self.setMapping(int(pathdata[28:]))

    def setIdx(self, idx):
        self.__Idx = idx

    def setType(self, type):
        self.__Type = type

    def setInsert(self, insert):
        self.__Insert = insert

    def setOverlap(self, overlap):
        self.__Overlap = overlap

    def setRelativePosition(self, relative):
        self.__RelativePosition = relative

    def setNextPath(self, path):
        self.__NextPath = path

    def setJump(self, jump):
        self.__Jump = jump

    def setPosition(self, position):
        self.__Position = (0x100000000 + position) & 0xFFFFFFFF

    def setSpeed(self, speed):
        self.__Speed = (0x10000 + speed) & 0xFFFF

    def setAcceleration(self, acceleration):
        self.__Acceleration = acceleration

    def setDeceleration(self, deceleration):
        self.__Deceleration = deceleration

    def setStopTime(self, stop):
        self.__StopTime = stop

    def setMapping(self, mapping):
        self.__Mapping = mapping

    def Idx(self):
        return self.__Idx

    def Type(self):
        return self.__Type

    def Insert(self):
        return self.__Insert

    def Overlap(self):
        return self.__Overlap

    def RelativePosition(self):
        return self.__RelativePosition

    def NextPath(self):
        return self.__NextPath

    def Jump(self):
        return self.__Jump

    def Position(self):
        return self.__Position

    def Speed(self):
        return self.__Speed

    def Acceleration(self):
        return self.__Acceleration

    def Deceleration(self):
        return self.__Deceleration

    def StopTime(self):
        return self.__StopTime

    def Mapping(self):
        return self.__Mapping


class DM2C_GoZero(object):
    def __init__(self, driver):
        self.update(driver)

    def update(self, driver):
        mode = int(driver.read(DM2C.GoZeroMode), 16)
        if mode & 0x01:
            self.setDirection(DM2C.Positive)
        else:
            self.setDirection(DM2C.Negative)

        if mode & 0x02:
            self.setMoveAfterGoZero(True)
        else:
            self.setMoveAfterGoZero(False)

        if mode & 0x04:
            self.setZero(DM2C.Origin)
        else:
            self.setZero(DM2C.Limit)

        self.setHighSpeed(int(driver.read(DM2C.GoZeroHighSpeed), 16))
        self.setLowSpeed(int(driver.read(DM2C.GoZeroLowSpeed), 16))
        self.setAcceleration(int(driver.read(DM2C.GoZeroAcceleration), 16))
        self.setDeceleration(int(driver.read(DM2C.GoZeroDeceleration), 16))
        self.setOvertravel(int(driver.read(DM2C.GoZeroOvertravel), 16))

    def setDirection(self, direction):
        self.__Direction = direction

    def setMoveAfterGoZero(self, move):
        self.__MoveAfterGoZero = move

    def setZero(self, zero):
        self.__Zero = zero

    def setHighSpeed(self, speed):
        self.__HighSpeed = speed

    def setLowSpeed(self, speed):
        self.__LowSpeed = speed

    def setAcceleration(self, acceleration):
        self.__Acceleration = acceleration

    def setDeceleration(self, deceleration):
        self.__Deceleration = deceleration

    def setOvertravel(self, overtravel):
        self.__OverTravel = overtravel

    def Direction(self):
        return self.__Direction

    def MoveAfterGoZero(self):
        return self.__MoveAfterGoZero

    def Zero(self):
        return self.__Zero

    def HighSpeed(self):
        return self.__HighSpeed

    def LowSpeed(self):
        return self.__LowSpeed

    def Acceleration(self):
        return self.__Acceleration

    def Deceleration(self):
        return self.__Deceleration

    def Overtravel(self):
        return self.__OverTravel


class DM2C_Jog(object):
    def __init__(self, driver):
        self.__Direction = DM2C.Positive
        self.__Cycle = 1
        self.update(driver)

    def update(self, driver):
        self.__Speed = int(driver.read(DM2C.JogSpeed), 16)
        self.__Interval = int(driver.read(DM2C.JogInterval), 16)
        # self.__Cycle=int(driver.read(DM2C.JogCycle),16)
        self.__Acceleration = int(driver.read(DM2C.JogAcceleration), 16)

    def setDirection(self, direction):
        self.__Direction = direction

    def setSpeed(self, speed):
        self.__Speed = speed

    def setInterval(self, interval):
        self.__Interval = interval

    # def setCycle(self,cycle):
    #     self.__Cycle=cycle

    def setAcceleration(self, acceleration):
        self.__Acceleration = acceleration

    def Direction(self):
        return self.__Direction

    def Speed(self):
        return self.__Speed

    def Interval(self):
        return self.__Interval

    def Cycle(self):
        return self.__Cycle

    def Acceleration(self):
        return self.__Acceleration


class DM2C_Status(object):
    def __init__(self, driver):
        self.update(driver)

    def update(self, driver):
        status = driver.read(DM2C.Status)
        if not status:
            self.__Connected = False
            status = "0000"
        else:
            self.__Connected = True
        status = int(status, 16)
        if status & 0x01:
            self.__Error = True
        else:
            self.__Error = False

        if status & 0x02:
            self.__Enable = True
        else:
            self.__Enable = False

        if status & 0x04:
            self.__Running = True
        else:
            self.__Running = False

        if status & 0x10:
            self.__OrderDone = True
        else:
            self.__OrderDone = False

        if status & 0x20:
            self.__PathDone = True
        else:
            self.__PathDone = False

        if status & 0x40:
            self.__GoZeroDone = True
        else:
            self.__GoZeroDone = False

    def Error(self):
        return self.__Error

    def Enable(self):
        return self.__Enable

    def Running(self):
        return self.__Running

    def OrderDone(self):
        return self.__OrderDone

    def PathDone(self):
        return self.__PathDone

    def GoZeroDone(self):
        return self.__GoZeroDone

    def Connected(self):
        return self.__Connected


class DM2C_Driver(object):
    def __init__(self, *args):
        self.__modbus = None
        self.__id = None
        self.__direction = None
        self.regAddress = ''
        self.regNumber = ''
        self.regData = ''
        self.__path = [None] * 16
        if args:
            for arg in args:
                if type(arg) == bytes:
                    self.setIdx(arg)
                elif type(arg) == ModbusRTU:
                    self.setModbus(arg)

    def connect(self, modbus, idx):
        self.setModbus(modbus)
        self.setIdx(idx)

    def setModbus(self, modbus):
        self.__modbus = modbus
        self.__modbus.closeCom()

    def setIdx(self, idx):
        self.__id = idx

    def reset(self):

        self.write(DM2C.ControlWord, DM2C.ControlWord_InitParameter)
        time.sleep(2)
        self.write(DM2C.ControlWord, DM2C.ControlWord_FactoryParameter)
        time.sleep(2)
        self.__direction = DM2C.Positive
        self.setDirection(DM2C.Positive)

    def stop(self):
        return self.write(DM2C.Trigger, DM2C.Trigger_Stop)

    def Position(self):
        position = int(self.read(DM2C.CurrentPositionH, 2), 16)
        return position if position < 0x8000 else position - 0x100000000 - 1

    def TargetPosition(self):
        return int(self.read(DM2C.TargetPositionH) + self.read(DM2C.CurrentPositionL), 16)

    def setZero(self):
        self.write(DM2C.Trigger, DM2C.Trigger_SetZero)

    def GoZeroSetting(self):
        self.__gozero = DM2C_GoZero(self)
        if self.read(DM2C.RunningDirection) == DM2C.RunningDirection_Positive:
            self.__direction = DM2C.Positive
        else:
            self.__direction = DM2C.Negative
        return copy.copy(self.__gozero)

    def goZero(self, *args):
        if args:
            gozero = args[0]
            self.write(DM2C.GoZeroMode,
                       "%04X" % (gozero.Direction() + (gozero.MoveAfterGoZero() << 1) + (gozero.Zero() << 2)))
            gozerodata = "%04X" % gozero.HighSpeed() + "%04X" % gozero.LowSpeed() + \
                         "%04X" % gozero.Acceleration() + "%04X" % gozero.Deceleration()
            self.write(DM2C.GoZeroHighSpeed, gozerodata)
            # if gozero.Overtravel() != self.__gozero.Overtravel():
            self.write(DM2C.GoZeroOvertravel, "%04X" % gozero.Overtravel())
            self.__gozero = copy.copy(gozero)

        self.write(DM2C.Trigger, DM2C.Trigger_GoZero)
        while not self.isGoZeroDone():
            time.sleep(0.25)

    def JogSetting(self):
        self.__jog = DM2C_Jog(self)
        if self.read(DM2C.RunningDirection) == DM2C.RunningDirection_Positive:
            self.__direction = DM2C.Positive
        else:
            self.__direction = DM2C.Negative
        return copy.copy(self.__jog)

    def jog(self, jog):
        if jog.Speed() != self.__jog.Speed():
            self.write(DM2C.JogSpeed, '%04x' % jog.Speed())
        if jog.Acceleration() != self.__jog.Acceleration():
            self.write(DM2C.JogAcceleration, '%04x' % jog.Acceleration())
        if jog.Interval() != self.__jog.Interval():
            self.write(DM2C.JogInterval, '%04x' % jog.Interval())
        if jog.Direction() == DM2C.Positive:
            d = DM2C.ControlWord_JogPositive
        else:
            d = DM2C.ControlWord_JogNegative
        for i in range(jog.Cycle()):
            self.write(DM2C.ControlWord, d)
        self.__jog = copy.copy(jog)

    def __newPath(self, idx):
        self.__path[idx] = DM2C_Path(self, idx)
        self.__path[idx].setIdx(idx)
        return self.__path[idx]

    def Path(self, *args):
        idx = -1
        if (len(args) != 1) or (args[0] not in range(16)):
            for i in range(16):
                if not self.__path[i]:
                    idx = i
                    return copy.copy(self.__newPath(idx))
        else:
            idx = args[0]
            if self.__path[idx]:
                return copy.copy(self.__path[idx])
            else:
                return copy.copy(self.__newPath(idx))

        if idx == -1:
            return None

    def addPath(self, path):
        idx = path.Idx()
        # self.write(DM2C.Path[idx], "%04X" %
        #            ((path.Type()) + (path.Insert() << 4) + (path.Overlap() << 5) +
        #             (path.RelativePosition() << 6) + (path.NextPath() << 8) + (path.Jump() << 14)))
        # self.write(DM2C.PathPositionH[idx], "%04X" % (path.Position() >> 16))
        # self.write(DM2C.PathPositionL[idx], "%04X" % (path.Position() & 0xFFFF))
        # self.write(DM2C.PathSpeed[idx], "%04X" % path.Speed())
        # self.write(DM2C.PathAcceleration[idx], "%04X" % path.Acceleration())
        # self.write(DM2C.PathDeceleration[idx], "%04X" % path.Deceleration())
        # self.write(DM2C.PathStopTime[idx], "%04X" % path.StopTime())
        # self.write(DM2C.PathMapping[idx], "%04X" % path.Mapping())

        setting = (path.Type()) + (path.Insert() << 4) + (path.Overlap() << 5) + \
                   (path.RelativePosition() << 6) + (path.NextPath() << 8) + (path.Jump() << 14)
        data = "%04X" % setting + "%08X" % path.Position() + "%04X" % path.Speed() + \
                "%04X" % path.Acceleration() + "%04X" % path.Deceleration() + "%04X" % path.StopTime() + "%04X" % path.Mapping()
        self.write(DM2C.Path[idx], data)

        self.__path[idx] = copy.copy(path)

    def deletePath(self, path):
        self.__path[path.Idx()] = None

    def goPath(self, path):
        # self.addPath(path)
        self.write(DM2C.Trigger, DM2C.Trigger_Path[path.Idx()])

    def Status(self):
        self.__status = DM2C_Status(self)
        return copy.copy(self.__status)

    def isConnected(self):
        return self.Status().Connected()

    def isEnable(self):
        return self.Status().Enable()

    def isPathDone(self):
        return self.Status().PathDone()

    def isRunning(self):
        return self.Status().Running()

    def isGoZeroDone(self):
        return self.Status().GoZeroDone()

    def setPulse(self, pulse):
        self.write(DM2C.Pulse, "%04X" % pulse)

    def Pulse(self):
        return int(self.read(DM2C.Pulse), 16)

    def setDirection(self, direction):
        if direction:
            self.write(DM2C.RunningDirection, DM2C.RunningDirection_Positive)
            self.__direction = DM2C.Positive
        else:
            self.write(DM2C.RunningDirection, DM2C.RunningDirection_Negative)
            self.__direction = DM2C.Negative

    def getError(self):
        return self.read(DM2C.Error)

    def comErr(self):
        print("Modbus: com error")
        self.__modbus.closeCom()
        return 0

    def comDone(self):
        self.__modbus.closeCom()
        return 1

    def read(self, address, number=1):
        while self.__modbus.isBusy():
            pass
        print("R: " + address)
        self.__modbus.setBusy(True)
        self.regAddress = address
        self.regNumber = '%04X' % number
        if self.readRegs():
            print("R: " + address + " " + self.regData)
            self.__modbus.setBusy(False)
            return self.regData
        self.__modbus.setBusy(False)
        return ''

    def write(self, address, data):
        number = int(len(data) / 4)
        while self.__modbus.isBusy():
            pass
        self.__modbus.setBusy(True)
        print("W: " + address + " " + data)
        self.regAddress = address
        self.regData = data
        if number == 1:
            self.writeReg()
            self.__modbus.setBusy(False)
        else:
            self.regNumber = '%04X' % number
            self.writeRegs()
            self.__modbus.setBusy(False)

    def readRegs(self):
        self.__modbus.address = self.__id
        self.__modbus.function = DM2C.ModbusFunctionCode_ReadRegs
        self.__modbus.data = bytes.fromhex(self.regAddress + self.regNumber)
        if not self.__modbus.tx():
            return self.comErr()
        if self.__modbus.rx(2) != self.__modbus.getMessage()[:2]:
            return self.comErr()
        bytenumber = int(self.regNumber, 16) * 2
        if self.__modbus.rx(1) != bytes([bytenumber]):
            return self.comErr()
        self.__modbus.data = bytes([bytenumber]) + self.__modbus.rx(bytenumber)
        self.__modbus.calcCRC()
        if self.__modbus.rx(2) != self.__modbus.getCRC():
            return self.comErr()
        self.regData = self.__modbus.data[1:].hex().upper()
        return self.comDone()

    def writeReg(self):
        self.__modbus.address = self.__id
        self.__modbus.function = DM2C.ModbusFunctionCode_WriteReg
        self.__modbus.data = bytes.fromhex(self.regAddress + self.regData)
        if not self.__modbus.tx():
            return self.comErr()
        if self.__modbus.rx(8) != self.__modbus.getMessage():
            return self.comErr()
        return self.comDone()

    def writeRegs(self):
        self.__modbus.address = self.__id
        self.__modbus.function = DM2C.ModbusFunctionCode_WriteRegs
        self.__modbus.data = bytes.fromhex(self.regAddress + self.regNumber) + \
                             bytes([int(len(self.regData) / 2)]) + \
                             bytes.fromhex(self.regData)
        if not self.__modbus.tx():
            return self.comErr()
        if self.__modbus.rx(6) != self.__modbus.getMessage()[:6]:
            return self.comErr()
        self.__modbus.data = self.__modbus.data[:4]
        self.__modbus.calcCRC()
        if self.__modbus.rx(2) != self.__modbus.getCRC():
            return self.comErr()
        return self.comDone()


if __name__ == '__main__':
    pass
