import sys
import os
import serial
from datetime import datetime

from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap, QIntValidator
from PySide6.QtCore import QTimer, Qt

from pypylon import pylon
import cv2

from dm2c import DM2C, DM2C_Driver, ModbusRTU
from Motor_Driver import Stage_LinearMovement

# 设置Qt平台插件路径
if hasattr(sys, '_MEIPASS'):
    plugin_path = os.path.join(sys._MEIPASS, 'PySide6', 'plugins', 'platforms')
else:
    plugin_path = os.path.join(os.path.dirname(QtCore.__file__), "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class SimpleCameraMotorController(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相机电机扫描系统")
        self.setGeometry(100, 100, 1000, 700)

        # 相机
        self.camera = None
        self.timer = QTimer(self)
        self.current_image = None

        # 电机
        self.motor = None
        self.modbus = None

        # 扫描相关
        self.is_scanning = False
        self.image_counter = 0
        self.save_folder = ""

        self.setup_ui()

    def setup_ui(self):
        """设置界面"""
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)

        # === 相机参数设置 ===
        control_panel.addWidget(QLabel("=== 相机参数 ==="))

        # 曝光模式
        control_panel.addWidget(QLabel("曝光模式:"))
        self.exposure_mode_combo = QComboBox()
        self.exposure_mode_combo.addItems(["Timed", "TriggerWidth"])
        control_panel.addWidget(self.exposure_mode_combo)

        # 曝光时间
        control_panel.addWidget(QLabel("曝光时间 (μs):"))
        self.exposure_time_edit = QLineEdit("10000")
        self.exposure_time_edit.setValidator(QIntValidator(30, 10000000))
        control_panel.addWidget(self.exposure_time_edit)

        # 自动曝光
        control_panel.addWidget(QLabel("自动曝光:"))
        self.auto_exposure_combo = QComboBox()
        self.auto_exposure_combo.addItems(["Off", "Once", "Continuous"])
        control_panel.addWidget(self.auto_exposure_combo)

        # 自动增益
        control_panel.addWidget(QLabel("自动增益:"))
        self.auto_gain_combo = QComboBox()
        self.auto_gain_combo.addItems(["Off", "Once", "Continuous"])
        control_panel.addWidget(self.auto_gain_combo)

        # 增益值
        control_panel.addWidget(QLabel("增益:"))
        self.gain_edit = QLineEdit("0")
        self.gain_edit.setValidator(QIntValidator(0, 23))
        control_panel.addWidget(self.gain_edit)

        # 设置参数按钮
        self.set_params_btn = QPushButton("设置相机参数")
        self.set_params_btn.clicked.connect(self.set_camera_parameters)
        control_panel.addWidget(self.set_params_btn)

        control_panel.addWidget(QLabel(""))  # 空行

        # === 相机控制 ===
        control_panel.addWidget(QLabel("=== 相机控制 ==="))

        self.open_camera_btn = QPushButton("打开相机")
        self.open_camera_btn.clicked.connect(self.open_camera)
        control_panel.addWidget(self.open_camera_btn)

        self.close_camera_btn = QPushButton("关闭相机")
        self.close_camera_btn.clicked.connect(self.close_camera)
        self.close_camera_btn.setEnabled(False)
        control_panel.addWidget(self.close_camera_btn)

        control_panel.addWidget(QLabel(""))

        # === 电机控制 ===
        control_panel.addWidget(QLabel("=== 电机控制 ==="))

        # COM端口
        com_layout = QHBoxLayout()
        com_layout.addWidget(QLabel("COM端口:"))
        self.com_port_edit = QLineEdit("COM7")
        com_layout.addWidget(self.com_port_edit)
        control_panel.addLayout(com_layout)

        # 电机步长设置
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("步长:"))
        self.step_size_edit = QLineEdit("10000")
        self.step_size_edit.setValidator(QIntValidator(1, 1000000))
        step_layout.addWidget(self.step_size_edit)
        control_panel.addLayout(step_layout)

        self.connect_motor_btn = QPushButton("连接电机")
        self.connect_motor_btn.clicked.connect(self.connect_motor)
        control_panel.addWidget(self.connect_motor_btn)

        control_panel.addWidget(QLabel(""))

        # === 扫描控制 ===
        control_panel.addWidget(QLabel("=== 扫描控制 ==="))

        self.start_scan_btn = QPushButton("开始扫描")
        self.start_scan_btn.clicked.connect(self.start_scan)
        self.start_scan_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        control_panel.addWidget(self.start_scan_btn)

        self.stop_scan_btn = QPushButton("停止扫描")
        self.stop_scan_btn.clicked.connect(self.stop_scan)
        self.stop_scan_btn.setEnabled(False)
        self.stop_scan_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        control_panel.addWidget(self.stop_scan_btn)

        # 单步执行按钮
        self.step_btn = QPushButton("走一步拍一张")
        self.step_btn.clicked.connect(self.step_and_capture)
        control_panel.addWidget(self.step_btn)

        control_panel.addWidget(QLabel(""))

        # === 状态显示 ===
        control_panel.addWidget(QLabel("=== 状态 ==="))

        self.status_label = QLabel("准备就绪")
        self.status_label.setWordWrap(True)
        control_panel.addWidget(self.status_label)

        self.counter_label = QLabel("已保存: 0 张")
        control_panel.addWidget(self.counter_label)

        control_panel.addStretch()

        # 右侧图像显示
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #ccc; }")
        self.image_label.setAlignment(Qt.AlignCenter)

        # 组装布局
        control_widget = QWidget()
        control_widget.setMaximumWidth(300)
        control_widget.setLayout(control_panel)

        main_layout.addWidget(control_widget)
        main_layout.addWidget(self.image_label, 1)

        self.setLayout(main_layout)

    def get_step_size(self):
        """获取步长设置"""
        try:
            return int(self.step_size_edit.text())
        except ValueError:
            return 10000  # 默认值

    def open_camera(self):
        """打开相机"""
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            # 设置默认参数
            self.set_default_parameters()

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)

            self.open_camera_btn.setEnabled(False)
            self.close_camera_btn.setEnabled(True)
            self.status_label.setText("相机已打开")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开相机失败: {str(e)}")

    def close_camera(self):
        """关闭相机"""
        if self.camera:
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
            self.camera = None

            self.open_camera_btn.setEnabled(True)
            self.close_camera_btn.setEnabled(False)
            self.status_label.setText("相机已关闭")

    def set_default_parameters(self):
        """设置默认相机参数"""
        try:
            # 设置为默认值
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.ExposureMode.SetValue("Timed")
            self.camera.ExposureTime.SetValue(10000)
            self.camera.GainAuto.SetValue("Off")
            self.camera.Gain.SetValue(0)

            # 更新UI显示
            self.exposure_mode_combo.setCurrentText("Timed")
            self.exposure_time_edit.setText("10000")
            self.auto_exposure_combo.setCurrentText("Off")
            self.auto_gain_combo.setCurrentText("Off")
            self.gain_edit.setText("0")

        except Exception as e:
            print(f"设置默认参数失败: {str(e)}")

    def set_camera_parameters(self):
        """设置相机参数"""
        if self.camera is None:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return

        try:
            # 设置自动曝光
            self.camera.ExposureAuto.SetValue(self.auto_exposure_combo.currentText())

            # 如果关闭自动曝光，则设置手动曝光参数
            if self.camera.ExposureAuto.GetValue() == "Off":
                self.camera.ExposureMode.SetValue(self.exposure_mode_combo.currentText())
                if self.camera.ExposureMode.GetValue() == "Timed":
                    exposure_time = int(self.exposure_time_edit.text())
                    self.camera.ExposureTime.SetValue(exposure_time)

            # 设置增益
            self.camera.GainAuto.SetValue(self.auto_gain_combo.currentText())
            if self.camera.GainAuto.GetValue() == "Off":
                gain_value = float(self.gain_edit.text())
                self.camera.Gain.SetValue(gain_value)

            self.status_label.setText("相机参数设置成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置参数失败: {str(e)}")
            self.status_label.setText(f"参数设置失败")

    def update_frame(self):
        """更新画面"""
        try:
            grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array
                self.current_image = img.copy()

                # 显示图像
                if len(img.shape) == 2:  # 灰度图
                    h, w = img.shape
                    q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
                else:  # 彩色图
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_img)
                scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)

                grab_result.Release()
        except:
            pass

    def connect_motor(self):
        """连接电机"""
        try:
            com_port = self.com_port_edit.text()
            self.modbus = ModbusRTU(serial.Serial(port=com_port, baudrate=115200, timeout=2, write_timeout=2))

            self.motor = Stage_LinearMovement(DM2C.Driver_01)
            self.motor.setModbus(self.modbus)
            self.motor.reset()
            self.motor.setRelativePositionPath(speed=800, acceleration=80, deceleration=80)

            self.connect_motor_btn.setText("电机已连接")
            self.connect_motor_btn.setEnabled(False)
            self.status_label.setText(f"电机已连接到 {com_port}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"连接电机失败: {str(e)}")

    def start_scan(self):
        """开始扫描"""
        if not self.camera:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return

        if not self.motor:
            QMessageBox.warning(self, "警告", "请先连接电机")
            return

        # 创建保存文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_folder = f"scan_{timestamp}"
        os.makedirs(self.save_folder, exist_ok=True)

        self.is_scanning = True
        self.image_counter = 0

        self.start_scan_btn.setEnabled(False)
        self.stop_scan_btn.setEnabled(True)

        step_size = self.get_step_size()
        self.status_label.setText(f"扫描中... (步长: {step_size})")

        # 开始循环扫描
        QTimer.singleShot(100, self.scan_loop)

    def scan_loop(self):
        """扫描循环"""
        if not self.is_scanning:
            return

        # 获取当前步长
        step_size = self.get_step_size()

        # 走一步
        try:
            self.motor.goRelativePosition(step_size)
            # 等待电机移动完成
            QTimer.singleShot(1000, self.capture_after_move)
        except Exception as e:
            self.status_label.setText(f"电机错误: {str(e)}")
            self.stop_scan()

    def capture_after_move(self):
        """移动后拍照"""
        if not self.is_scanning:
            return

        # 拍照保存
        if self.current_image is not None:
            filename = os.path.join(self.save_folder, f"{self.image_counter:04d}.png")
            cv2.imwrite(filename, self.current_image)
            self.image_counter += 1
            self.counter_label.setText(f"已保存: {self.image_counter} 张")

        # 继续下一步
        QTimer.singleShot(100, self.scan_loop)

    def stop_scan(self):
        """停止扫描"""
        self.is_scanning = False

        if self.motor:
            try:
                self.motor.stop()
            except:
                pass

        self.start_scan_btn.setEnabled(True)
        self.stop_scan_btn.setEnabled(False)
        self.status_label.setText(f"扫描停止，共保存 {self.image_counter} 张")

    def step_and_capture(self):
        """单步执行：走一步拍一张"""
        if not self.camera:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return

        if not self.motor:
            QMessageBox.warning(self, "警告", "请先连接电机")
            return

        # 获取步长
        step_size = self.get_step_size()

        try:
            # 移动电机
            self.motor.goRelativePosition(step_size)
            self.status_label.setText(f"电机移动中... (步长: {step_size})")

            # 等待1秒后拍照
            QTimer.singleShot(1000, self.single_capture)

        except Exception as e:
            self.status_label.setText(f"错误: {str(e)}")

    def single_capture(self):
        """单次拍照保存"""
        if self.current_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"single_{timestamp}.png"
            cv2.imwrite(filename, self.current_image)
            self.status_label.setText(f"已保存: {filename}")

    def closeEvent(self, event):
        """关闭窗口"""
        self.stop_scan()
        self.close_camera()
        if self.modbus:
            self.modbus.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleCameraMotorController()
    window.show()
    sys.exit(app.exec())