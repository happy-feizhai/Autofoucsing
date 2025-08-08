import sys
import os
import cv2
import numpy as np
from numba import njit
import queue
from dataclasses import dataclass
import time, threading
import concurrent.futures
from typing import List, Tuple, Optional
from Motor_Driver import Stage_LinearMovement
from dm2c import DM2C, DM2C_Driver, ModbusRTU
import serial
# PySide6 相关导入
from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap, QIntValidator, QDoubleValidator
from PySide6.QtCore import QTimer, Qt

# Basler相机相关导入
from pypylon import pylon

# 设置Qt平台插件路径
if hasattr(sys, '_MEIPASS'):
    plugin_path = os.path.join(sys._MEIPASS, 'PySide6', 'plugins', 'platforms')
else:
    plugin_path = os.path.join(os.path.dirname(QtCore.__file__), "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


# ============窗口滑动类===========
# 在文件开头，QTimer, Qt 导入之后添加这个类：

class ClickableImageLabel(QLabel):
    """可点击的图像标签，支持鼠标长按事件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.setMouseTracking(True)
        self.is_pressed = False

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton and self.parent_viewer:
            # 获取点击位置
            pos = event.pos()

            # 转换为图像坐标
            label_size = self.size()
            pixmap = self.pixmap()

            if pixmap:
                # 计算图像在标签中的实际位置和大小
                pixmap_size = pixmap.size()

                # 计算缩放比例
                scale_x = pixmap_size.width() / label_size.width()
                scale_y = pixmap_size.height() / label_size.height()
                scale = max(scale_x, scale_y)

                # 计算实际显示大小
                display_width = pixmap_size.width() / scale
                display_height = pixmap_size.height() / scale

                # 计算偏移量（居中显示）
                offset_x = (label_size.width() - display_width) / 2
                offset_y = (label_size.height() - display_height) / 2

                # 转换点击坐标到图像坐标
                img_x = (pos.x() - offset_x) * scale
                img_y = (pos.y() - offset_y) * scale

                # 确保坐标在有效范围内
                if 0 <= img_x <= pixmap_size.width() and 0 <= img_y <= pixmap_size.height():
                    self.parent_viewer.start_mouse_control(img_x, img_y)
                    self.is_pressed = True

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.is_pressed:
            self.is_pressed = False
            if self.parent_viewer:
                self.parent_viewer.stop_mouse_control()

    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 可选，用于拖动时实时更新"""
        if self.is_pressed and self.parent_viewer:
            pos = event.pos()

            # 转换为图像坐标（使用相同的转换逻辑）
            label_size = self.size()
            pixmap = self.pixmap()

            if pixmap:
                pixmap_size = pixmap.size()
                scale_x = pixmap_size.width() / label_size.width()
                scale_y = pixmap_size.height() / label_size.height()
                scale = max(scale_x, scale_y)

                display_width = pixmap_size.width() / scale
                display_height = pixmap_size.height() / scale

                offset_x = (label_size.width() - display_width) / 2
                offset_y = (label_size.height() - display_height) / 2

                img_x = (pos.x() - offset_x) * scale
                img_y = (pos.y() - offset_y) * scale

                if 0 <= img_x <= pixmap_size.width() and 0 <= img_y <= pixmap_size.height():
                    # 先停止当前移动
                    self.parent_viewer.stop_mouse_control()
                    # 开始新的移动
                    self.parent_viewer.start_mouse_control(img_x, img_y)


# ========== PID控制器类 ==========
class PIDController:
    """PID控制器"""

    def __init__(self, Kp=0.5, Ki=0.0, Kd=0.1):
        self.Kp = Kp  # 比例系数
        self.Ki = Ki  # 积分系数
        self.Kd = Kd  # 微分系数

        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        """更新PID控制器，返回控制输出"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            dt = 0.001  # 防止除零

        # 比例项
        P = self.Kp * error

        # 积分项
        self.integral += error * dt
        I = self.Ki * self.integral

        # 微分项
        derivative = (error - self.previous_error) / dt
        D = self.Kd * derivative

        # 总输出
        output = P + I + D

        # 更新状态
        self.previous_error = error
        self.last_time = current_time

        return output

    def reset(self):
        """重置PID控制器"""
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

# ========== 电机控制接口类 ==========
class MotorController:
    """电机控制接口类 - 需要用户根据实际电机实现这些方法"""

    def __init__(self):

        # 电机参数配置
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


        self.serialport = "COM7"
        self.modbus = ModbusRTU(serial.Serial(port=self.serialport, baudrate=115200, timeout=2, write_timeout=2))
        self.xaxis = Stage_LinearMovement(DM2C.Driver_01)
        self.xaxis.setModbus(self.modbus)
        self.xaxis.reset()
        self.xaxis.setRelativePositionPath(speed=800, acceleration=80, deceleration=80)
        self.xaxis.setAbsolutePositionPath()
        self.xaxis.setSpeedPath()
        self.xaxis.setJogSpeed(Device_Stage.X_LowSpeed, Device_Stage.X_HighSpeed)

        self.zaxis = Stage_LinearMovement(DM2C.Driver_03)
        self.zaxis.setModbus(self.modbus)
        self.zaxis.reset()
        self.zaxis.setRelativePositionPath()
        self.zaxis.setAbsolutePositionPath()
        self.zaxis.setSpeedPath()
        self.zaxis.setJogSpeed(Device_Stage.X_LowSpeed, Device_Stage.X_HighSpeed)

    def get_x_position(self):
        """获取X轴电机当前位置（单位：mm）
        实际使用时，请替换为真实的电机位置读取函数
        """
        return self.xaxis.Position()

    def get_y_position(self):
        """获取Y轴电机当前位置（单位：mm）
        实际使用时，请替换为真实的电机位置读取函数
        """
        return self.zaxis.Position()

    def move_x_to_absolute(self, position):
        """移动X轴电机到绝对位置（单位：mm）
        实际使用时，请替换为真实的电机移动函数
        """
        print(f"Moving X motor to absolute position: {position:.3f} mm")
        self.xaxis.goAbsolutePosition(position)

    def move_y_to_absolute(self, position):
        """移动Y轴电机到绝对位置（单位：mm）
        实际使用时，请替换为真实的电机移动函数
        """
        print(f"Moving Y motor to absolute position: {position:.3f} mm")
        self.zaxis.goAbsolutePosition(position)

    def move_x_to_relative(self, position):
        """移动x轴电机到相对位置（单位：mm）
        """
        print(f"Moving X motor : {position:.3f}mm")
        self.xaxis.goRelativePosition(position)

    def move_y_to_relative(self, position):
        """移动y轴电机到相对位置（单位：mm）
        """
        print(f"Moving X motor : {position:.3f}mm")
        self.zaxis.goRelativePosition(position)

    def move_x_go_speed(self, speed):
        """驱动x电机按指定速度移动
        """
        print(f"Moving X motor at speed: {speed}")
        self.xaxis.goSpeed(speed)

    def move_y_go_speed(self, speed):
        """驱动y电机按指定速度移动
        """
        print(f"Moving X motor at speed: {speed}")
        self.zaxis.goSpeed(speed)


    def stop_x(self):
        """
        X轴电机急停
        """
        self.xaxis.stop()
        print("Stopping X motor")


    def stop_y(self):
        """
        Y轴电机急停
        """
        self.zaxis.stop()
        print("Stopping Y motor")


# ========== 瞳孔检测相关函数 ==========
@njit
def compute_image_sharpness_numba(img: np.ndarray) -> float:
    """使用 Numba 加速计算图像清晰度"""
    h, w = img.shape
    F = 0

    for x in range(1, w - 1, 2):
        for y in range(1, h - 1, 2):
            # 计算 Pave(x, y)
            pave = (img[y, x] + img[y, x + 1] + img[y, x - 1] + img[y + 1, x] + img[y - 1, x]) / 5

            # 计算 G1st(x, y)
            g1st = (abs(img[y, x + 1] - pave) + abs(img[y + 1, x] - pave) + abs(img[y + 1, x + 1] - pave)) ** 2

            # 计算 G2nd(x, y)
            g2nd = (abs(img[y, x + 2] - pave) + abs(img[y + 2, x] - pave) + abs(img[y + 2, x + 2] - pave)) ** 2

            # 计算清晰度 F
            F += g1st * g2nd
    F = F * 1e-6

    return F


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """图像预处理：增强对比度和降噪"""
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # 2. 高斯滤波降噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return denoised


def evaluate_contour_quality(contour: np.ndarray) -> float:
    """评估轮廓质量，返回得分"""
    # 1. 圆形度评估
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    return circularity


def evaluate_circle_quality(img: np.ndarray, circle: Tuple[int, int, int]) -> float:
    """优化的圆质量评估函数 - 修正版"""
    x, y, r = circle
    h, w = img.shape

    # 边界检查
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return 0.0

    # 方法1: 使用ROI（感兴趣区域）来减少计算量
    # 先裁剪出包含圆及其周围的区域
    outer_radius = min(int(r * 1.8), min(w, h) // 2)

    # 计算ROI边界
    roi_x1 = max(0, x - outer_radius)
    roi_y1 = max(0, y - outer_radius)
    roi_x2 = min(w, x + outer_radius + 1)
    roi_y2 = min(h, y + outer_radius + 1)

    # 裁剪ROI
    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_h, roi_w = roi.shape

    # 在ROI中的新圆心位置
    roi_cx = x - roi_x1
    roi_cy = y - roi_y1

    # 创建ROI大小的mask（更小的内存占用）
    inner_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.circle(inner_mask, (roi_cx, roi_cy), r, 255, -1)

    # 创建外环mask
    outer_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.circle(outer_mask, (roi_cx, roi_cy), min(outer_radius, roi_w // 2, roi_h // 2), 255, -1)
    cv2.circle(outer_mask, (roi_cx, roi_cy), r, 0, -1)



    # 计算均值（在小的ROI上计算，更快）
    inner_mean = cv2.mean(roi, inner_mask)[0]
    outer_mean = cv2.mean(roi, outer_mask)[0]

    # 对比度得分（瞳孔应该比周围区域更暗）
    contrast = outer_mean - inner_mean

    return max(0, contrast)


def process_single_threshold(args: Tuple[np.ndarray, int, int]) -> Tuple[Optional[Tuple[int, int, int]], float]:
    """处理单个阈值的函数，用于并行处理"""
    processed, thresh_val, r_threshold = args

    # 阈值化
    _, fixed_thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作
    start = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(fixed_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    end = time.perf_counter()
    # print(f"{thresh_val}形态学执行时间: {(end - start) * 1000:.2f} ms")


    best_circle = None
    best_score = 0

    # 查找轮廓
    start = time.perf_counter()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    end = time.perf_counter()
    # print(f"{thresh_val}查找轮廓执行时间: {(end - start) * 1000:.2f} ms")


    for contour in contours:

        area = cv2.contourArea(contour)
        if area < 55000:  # 太小的轮廓跳过
            continue

        # 拟合圆
        start = time.perf_counter()
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), int(radius)
        end = time.perf_counter()
        # print(f"{thresh_val}拟合圆执行时间: {(end - start) * 1000:.2f} ms")

        if radius < r_threshold or radius > 2 * r_threshold:  # 半径范围检查
            continue

        # 计算轮廓质量得分

        start = time.perf_counter()
        contour_score = evaluate_contour_quality(contour)
        end = time.perf_counter()
        # print(f"{thresh_val}评分执行时间: {(end - start) * 1000:.2f} ms")


        # 计算灰度对比度得分
        # contrast_score = evaluate_circle_quality(processed, (cx, cy, radius))
        #
        # # 综合得分
        # total_score = contour_score * 0.6 + contrast_score * 0.4
        total_score = contour_score

        if total_score > best_score:
            best_score = total_score
            best_circle = (cx, cy, radius)

    return best_circle, best_score


def detect_pupil_contour_parallel(img: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """使用多进程并行处理的轮廓检测"""
    # 预处理
    # processed = preprocess_image(img)

    processed = preprocess_image(img)
    r_threshold = 150

    thresh_values = [30, 40, 50, 60]
    args_list = [(processed, thresh_val, r_threshold) for thresh_val in thresh_values]

    best_circle = None
    best_score = 0

    # 改用进程池
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_threshold, args) for args in args_list]

        for future in concurrent.futures.as_completed(futures):
            circle, score = future.result()
            if score > best_score:
                best_score = score
                best_circle = circle

    return best_circle

def detect_pupil_contour(img: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """使用轮廓检测瞳孔"""
    # 预处理
    start = time.perf_counter()
    processed = preprocess_image(img)
    end = time.perf_counter()
    # print(f"预处理执行时间: {(end - start) * 1000:.2f} ms")


    r_threshold = 150

    # 多种阈值方法
    thresholds = []

    # 固定阈值（多个值）

    # for thresh_val in [30, 40, 50, 60]:
    thresh_val = 30
    start = time.perf_counter()
    _, fixed_thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)
    thresholds.append(fixed_thresh)
    end = time.perf_counter()
    # print(f"{thresh_val}阈值分割执行时间: {(end - start) * 1000:.2f} ms")

    best_circle = None
    best_score = 0

    for thresh in thresholds:
        # 形态学操作
        start = time.perf_counter()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        end = time.perf_counter()
        # print(f"{thresh_val}形态学执行时间: {(end - start) * 1000:.2f} ms")


        # 查找轮廓
        start = time.perf_counter()
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        end = time.perf_counter()
        # print(f"{thresh_val}查找轮廓执行时间: {(end - start) * 1000:.2f} ms")

        # 早期剔除：先按面积排序，只处理最大的N个轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_contours_to_process = 5  # 只处理前5个最大的轮廓

        for i, contour in enumerate(contours):
            if i >= max_contours_to_process:
                break

            area = cv2.contourArea(contour)
            if area < 55000:  # 太小的轮廓跳过
                continue

            # 拟合圆
            start = time.perf_counter()
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), int(radius)
            end = time.perf_counter()
            # print(f"{thresh_val}拟合圆执行时间: {(end - start) * 1000:.2f} ms")


            if radius < r_threshold or radius > 2.5 * r_threshold:  # 半径范围检查
                continue

            # 计算轮廓质量得分
            contour_score = evaluate_contour_quality(contour)

            # 计算灰度对比度得分
            # contrast_score = evaluate_circle_quality(img, (cx, cy, radius))

            # 综合得分
            # total_score = contour_score * 0.6 + contrast_score * 0.4

            total_score = contour_score
            if total_score > best_score:
                best_score = total_score
                best_circle = (cx, cy, radius)

    return best_circle


def robust_pupil_detection(img: np.ndarray) -> Tuple[Optional[Tuple[int, int, int]], float]:
    """
    鲁棒的瞳孔检测主函数
    返回: (瞳孔圆形参数, 清晰度值)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用轮廓检测
    detected_circle = detect_pupil_contour(img)

    # detected_circle = detect_pupil_contour_parallel(img)

    if detected_circle is None:
        return None, 0.0

    # 提取瞳孔区域并计算清晰度
    x, y, radius = detected_circle

    # 扩展区域用于清晰度计算
    expand_factor = 1.8
    expanded_radius = int(radius * expand_factor)

    # 边界检查
    h, w = img.shape
    x1 = max(0, x - expanded_radius)
    y1 = max(0, y - expanded_radius)
    x2 = min(w, x + expanded_radius)
    y2 = min(h, y + expanded_radius)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None, 0.0

    # 计算清晰度
    start = time.perf_counter()
    sharpness = compute_image_sharpness_numba(crop)
    end = time.perf_counter()
    # print(f"清晰度执行时间: {(end - start) * 1000:.2f} ms")

    # sharpness = 0

    return detected_circle, sharpness

@dataclass
class DetectionResult:
    """检测结果数据类"""
    pupil_circle: Optional[Tuple[int, int, int]]
    sharpness: float
    image: np.ndarray
    timestamp: float

# ========== 主界面类 ==========
class PupilCameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("瞳孔识别相机系统 - Pupil Detection Camera System")
        self.setGeometry(100, 100, 1200, 800)

        # 相机相关变量
        self.camera = None
        self.timer = QTimer(self)
        self.pupil_detection_mode = False  # 瞳孔检测模式标志
        self.pupil_alignment_mode = False  # 瞳孔对齐模式标志

        # 瞳孔位置相关
        self.current_pupil_position = None  # 当前瞳孔位置 (x, y)
        self.target_pupil_position = (1024, 1024)  # 理论瞳孔位置（默认图像中心）
        self.pixel_to_mm_ratio = 0.0106  # 像素到毫米的转换比例（需要根据实际系统标定）

        # PID控制器
        self.pid_x = PIDController(Kp=1, Ki=0.2, Kd=0.5)
        self.pid_y = PIDController(Kp=1, Ki=0.2, Kd=0.5)

        # 电机控制器
        self.motor_controller = None

        # 对齐状态
        self.alignment_tolerance = 30  # 对齐容差（像素）
        self.x_aligned = False
        self.y_aligned = False

        # ============ 非阻塞检测相关 ============
        self.detection_thread = None
        self.detection_queue = queue.Queue(maxsize=2)  # 最多缓存2个结果
        self.detection_lock = threading.Lock()
        self.is_detecting = False
        self.stop_detection = False
        self.pupil_alignment_lock = threading.Lock()

        # 电机控制线程池
        self.motor_threads = []
        self.motor_lock = threading.Lock()


        # 性能统计
        self.frame_count = 0
        self.detection_times = []
        self.last_detection_time = 0

        # 添加图像队列用于传递给检测线程
        self.image_queue = queue.Queue(maxsize=2)
        self.latest_image = None
        self.image_lock = threading.Lock()

        # 创建UI
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = QVBoxLayout()
        control_panel.setSpacing(10)

        # 相机参数控制
        control_panel.addWidget(QLabel("=== Camera Parameters Setup ==="))

        # 曝光模式
        self.exposure_mode_label = QLabel("Exposure Mode:")
        self.exposure_mode_combo = QComboBox()
        self.exposure_mode_combo.addItems(["Timed", "TriggerWidth"])
        control_panel.addWidget(self.exposure_mode_label)
        control_panel.addWidget(self.exposure_mode_combo)

        # 曝光时间
        self.exposure_time_label = QLabel("Exposure Time (us):")
        self.exposure_time_edit = QLineEdit("10000")
        self.exposure_time_edit.setValidator(QIntValidator(30, 10000000))
        control_panel.addWidget(self.exposure_time_label)
        control_panel.addWidget(self.exposure_time_edit)

        # 自动曝光
        self.auto_exposure_label = QLabel("Auto Exposure:")
        self.auto_exposure_combo = QComboBox()
        self.auto_exposure_combo.addItems(["Off", "Once", "Continuous"])
        control_panel.addWidget(self.auto_exposure_label)
        control_panel.addWidget(self.auto_exposure_combo)

        # 自动增益
        self.auto_gain_label = QLabel("Auto Gain:")
        self.auto_gain_combo = QComboBox()
        self.auto_gain_combo.addItems(["Off", "Once", "Continuous"])
        control_panel.addWidget(self.auto_gain_label)
        control_panel.addWidget(self.auto_gain_combo)

        # 增益值
        self.gain_label = QLabel("Gain:")
        self.gain_edit = QLineEdit("0")
        self.gain_edit.setValidator(QIntValidator(0, 23))
        control_panel.addWidget(self.gain_label)
        control_panel.addWidget(self.gain_edit)

        # 设置参数按钮
        self.set_params_button = QPushButton("Set Parameters")
        self.set_params_button.clicked.connect(self.set_camera_parameters)
        control_panel.addWidget(self.set_params_button)

        control_panel.addWidget(QLabel(""))  # 空行

        # 相机控制按钮
        control_panel.addWidget(QLabel("=== Camera Control ==="))

        self.open_button = QPushButton("Open Camera")
        self.open_button.clicked.connect(self.start_camera)
        control_panel.addWidget(self.open_button)

        self.close_button = QPushButton("Close Camera")
        self.close_button.clicked.connect(self.close_camera)
        control_panel.addWidget(self.close_button)

        self.connect_motor_button = QPushButton("Connect Motor")
        self.connect_motor_button.clicked.connect(self.connect_motor)
        control_panel.addWidget(self.connect_motor_button)

        # 瞳孔检测按钮
        self.pupil_detect_button = QPushButton("Start Pupil Detection")
        self.pupil_detect_button.clicked.connect(self.toggle_pupil_detection)
        self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        control_panel.addWidget(self.pupil_detect_button)

        # 瞳孔对齐按钮
        self.pupil_align_button = QPushButton("Start Pupil Alignment")
        self.pupil_align_button.clicked.connect(self.toggle_pupil_alignment)
        self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        self.pupil_align_button.setEnabled(False)  # 初始状态禁用
        control_panel.addWidget(self.pupil_align_button)

        control_panel.addWidget(QLabel(""))  # 空行

        # 对齐参数设置
        control_panel.addWidget(QLabel("=== Alignment Settings ==="))

        # 目标位置设置
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target X:"))
        self.target_x_edit = QLineEdit(str(self.target_pupil_position[0]))
        self.target_x_edit.setValidator(QIntValidator(0, 9999))
        self.target_x_edit.textChanged.connect(self.update_target_position)
        target_layout.addWidget(self.target_x_edit)
        target_layout.addWidget(QLabel("Y:"))
        self.target_y_edit = QLineEdit(str(self.target_pupil_position[1]))
        self.target_y_edit.setValidator(QIntValidator(0, 9999))
        self.target_y_edit.textChanged.connect(self.update_target_position)
        target_layout.addWidget(self.target_y_edit)
        control_panel.addLayout(target_layout)

        # PID参数设置
        pid_layout = QGridLayout()
        pid_layout.addWidget(QLabel("PID:"), 0, 0)
        pid_layout.addWidget(QLabel("Kp"), 0, 1)
        pid_layout.addWidget(QLabel("Ki"), 0, 2)
        pid_layout.addWidget(QLabel("Kd"), 0, 3)

        # X轴PID
        pid_layout.addWidget(QLabel("X:"), 1, 0)
        self.pid_x_kp = QLineEdit("0.5")
        self.pid_x_kp.setValidator(QDoubleValidator(0, 10, 3))
        pid_layout.addWidget(self.pid_x_kp, 1, 1)
        self.pid_x_ki = QLineEdit("0.0")
        self.pid_x_ki.setValidator(QDoubleValidator(0, 10, 3))
        pid_layout.addWidget(self.pid_x_ki, 1, 2)
        self.pid_x_kd = QLineEdit("0.1")
        self.pid_x_kd.setValidator(QDoubleValidator(0, 10, 3))
        pid_layout.addWidget(self.pid_x_kd, 1, 3)

        # Y轴PID
        pid_layout.addWidget(QLabel("Y:"), 2, 0)
        self.pid_y_kp = QLineEdit("0.5")
        self.pid_y_kp.setValidator(QDoubleValidator(0, 10, 3))
        pid_layout.addWidget(self.pid_y_kp, 2, 1)
        self.pid_y_ki = QLineEdit("0.0")
        self.pid_y_ki.setValidator(QDoubleValidator(0, 10, 3))
        pid_layout.addWidget(self.pid_y_ki, 2, 2)
        self.pid_y_kd = QLineEdit("0.1")
        self.pid_y_kd.setValidator(QDoubleValidator(0, 10, 3))
        pid_layout.addWidget(self.pid_y_kd, 2, 3)

        control_panel.addLayout(pid_layout)

        # 更新PID参数按钮
        self.update_pid_button = QPushButton("Update PID Parameters")
        self.update_pid_button.clicked.connect(self.update_pid_parameters)
        control_panel.addWidget(self.update_pid_button)

        control_panel.addWidget(QLabel(""))  # 空行

        # 检测结果显示
        control_panel.addWidget(QLabel("=== Result ==="))

        # 清晰度显示
        self.sharpness_label = QLabel("Sharpness:")
        control_panel.addWidget(self.sharpness_label)

        self.sharpness_text = QTextEdit()
        self.sharpness_text.setMaximumHeight(60)
        self.sharpness_text.setReadOnly(True)
        control_panel.addWidget(self.sharpness_text)

        # 瞳孔信息显示
        self.pupil_info_label = QLabel("Pupil Info:")
        control_panel.addWidget(self.pupil_info_label)

        self.pupil_info_text = QTextEdit()
        self.pupil_info_text.setMaximumHeight(100)
        self.pupil_info_text.setReadOnly(True)
        control_panel.addWidget(self.pupil_info_text)

        # 对齐状态显示
        self.alignment_status_label = QLabel("Alignment Status:")
        control_panel.addWidget(self.alignment_status_label)

        self.alignment_status_text = QTextEdit()
        self.alignment_status_text.setMaximumHeight(80)
        self.alignment_status_text.setReadOnly(True)
        control_panel.addWidget(self.alignment_status_text)

        control_panel.addStretch()  # 添加弹性空间

        # 右侧图像显示区域
        image_layout = QVBoxLayout()

        self.image_label = ClickableImageLabel(parent=self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #ccc; }")
        self.image_label.setMinimumSize(640, 480)
        image_layout.addWidget(self.image_label)

        # 组合布局
        control_widget = QWidget()
        control_widget.setMaximumWidth(400)
        control_widget.setLayout(control_panel)

        main_layout.addWidget(control_widget)
        main_layout.addLayout(image_layout, 1)

        self.setLayout(main_layout)

    def update_target_position(self):
        """更新目标瞳孔位置"""
        try:
            x = int(self.target_x_edit.text())
            y = int(self.target_y_edit.text())
            self.target_pupil_position = (x, y)
        except ValueError:
            pass

    def update_pid_parameters(self):
        """更新PID参数"""
        try:
            # 更新X轴PID
            self.pid_x.Kp = float(self.pid_x_kp.text())
            self.pid_x.Ki = float(self.pid_x_ki.text())
            self.pid_x.Kd = float(self.pid_x_kd.text())

            # 更新Y轴PID
            self.pid_y.Kp = float(self.pid_y_kp.text())
            self.pid_y.Ki = float(self.pid_y_ki.text())
            self.pid_y.Kd = float(self.pid_y_kd.text())

            self.sharpness_text.setText("PID参数已更新")
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的PID参数")
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

            self.sharpness_text.setText("参数设置成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置参数失败: {str(e)}")
    def start_camera(self):
        """打开相机 - 修改版"""
        try:
            if self.camera is not None:
                self.camera.StopGrabbing()
                self.camera.Close()

            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # 启动定时器更新画面
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(20)  # 20ms更新一次（50 FPS）

            self.sharpness_text.setText("相机已打开")
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开相机: {str(e)}")

    def close_camera(self):
        """关闭相机 - 修改版"""
        if self.camera is not None:
            # 停止检测线程
            self.stop_detection_thread()

            # 停止定时器和相机
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
            self.camera = None

            # 等待所有电机线程完成
            for thread in self.motor_threads:
                if thread.is_alive():
                    thread.join(timeout=1)
            self.motor_threads.clear()

            # 重置UI状态
            self.sharpness_text.setText("相机已关闭")
            self.open_button.setEnabled(True)
            self.close_button.setEnabled(False)
            self.pupil_detection_mode = False
            self.pupil_alignment_mode = False
            self.pupil_detect_button.setText("开始瞳孔检测")
            self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
            self.pupil_align_button.setText("开始瞳孔对齐")
            self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
            self.pupil_align_button.setEnabled(False)

    def connect_motor(self):
        """连接电机"""
        if self.motor_controller is None:
            # 电机控制器
            try:
                self.motor_controller = MotorController()
                self.connect_motor_button.setEnabled(False)
            except Exception as e:
                print(e)
                self.connect_motor_button.setEnabled(True)



    def toggle_pupil_detection(self):
        """切换瞳孔检测模式 - 修改版"""
        if self.camera is None:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return

        self.pupil_detection_mode = not self.pupil_detection_mode

        if self.pupil_detection_mode:
            # 启动检测线程
            self.start_detection_thread()

            self.pupil_detect_button.setText("停止瞳孔检测")
            self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
            self.sharpness_text.setText("瞳孔检测模式已开启（非阻塞）")
            self.pupil_align_button.setEnabled(True)
        else:
            # 停止检测线程
            self.stop_detection_thread()

            self.pupil_detect_button.setText("开始瞳孔检测")
            self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
            self.sharpness_text.setText("瞳孔检测模式已关闭")
            self.pupil_info_text.clear()
            self.pupil_align_button.setEnabled(False)

            if self.pupil_alignment_mode:
                self.toggle_pupil_alignment()

    def start_detection_thread(self):
        """启动检测线程"""
        if self.detection_thread and self.detection_thread.is_alive():
            return  # 线程已在运行

        self.stop_detection = False
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()

    def stop_detection_thread(self):
        """停止检测线程"""
        self.stop_detection = True
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1)

    def detection_worker(self):
        """检测工作线程 - 不直接访问相机"""
        while not self.stop_detection:
            try:
                # 从队列获取图像，而不是直接从相机获取
                img = self.image_queue.get(timeout=0.1)  # 100ms超时

                # 标记正在检测
                with self.detection_lock:
                    self.is_detecting = True

                # 执行瞳孔检测
                start_time = time.perf_counter()
                pupil_circle, sharpness = robust_pupil_detection(img)
                detection_time = (time.perf_counter() - start_time) * 1000

                # 创建结果
                result = DetectionResult(
                    pupil_circle=pupil_circle,
                    sharpness=sharpness,
                    image=img,
                    timestamp=time.time()
                )

                # 将结果放入队列
                try:
                    if self.detection_queue.full():
                        self.detection_queue.get_nowait()
                    self.detection_queue.put_nowait(result)
                except queue.Full:
                    pass

                # 更新统计
                with self.detection_lock:
                    self.is_detecting = False
                    self.last_detection_time = detection_time
                    self.detection_times.append(detection_time)
                    if len(self.detection_times) > 100:
                        self.detection_times.pop(0)

            except queue.Empty:
                # 没有新图像，继续等待
                pass
            except Exception as e:
                print(f"检测线程错误: {e}")
                with self.detection_lock:
                    self.is_detecting = False
                time.sleep(0.1)

    def toggle_pupil_alignment(self):
        """切换瞳孔对齐模式"""
        if not self.pupil_detection_mode:
            QMessageBox.warning(self, "警告", "请先开启瞳孔检测模式")
            return

        self.pupil_alignment_mode = not self.pupil_alignment_mode

        if self.pupil_alignment_mode:
            # 重置PID控制器
            self.pid_x.reset()
            self.pid_y.reset()
            self.x_aligned = False
            self.y_aligned = False

            self.pupil_align_button.setText("停止瞳孔对齐")
            self.pupil_align_button.setStyleSheet("QPushButton { background-color: #ff9800; color: white; }")
            self.alignment_status_text.setText("瞳孔对齐模式已开启")
        else:
            self.pupil_align_button.setText("开始瞳孔对齐")
            self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
            self.alignment_status_text.setText("瞳孔对齐模式已关闭")

    def perform_alignment_nonblocking(self, pupil_x, pupil_y):
        """非阻塞的瞳孔对齐控制"""
        target_x, target_y = self.target_pupil_position

        # 计算像素偏差
        error_x = target_x - pupil_x
        error_y = target_y - pupil_y
        print(f"x方向偏差:{error_x} 像素"
              f"y方向变差:{error_y} 像素")

        # 检查对齐状态
        self.x_aligned = abs(error_x) < self.alignment_tolerance
        self.y_aligned = abs(error_y) < self.alignment_tolerance

        if self.x_aligned and self.y_aligned:
            self.alignment_status_text.setText(
                f"对齐成功！\n"
                f"X偏差: {error_x:.1f} 像素\n"
                f"Y偏差: {error_y:.1f} 像素\n"
                f"状态: 已对准"
            )
            self.pupil_alignment_mode = not self.pupil_alignment_mode
            self.pupil_align_button.setText("开始瞳孔对齐")
            self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
            return

        # 清理已完成的线程
        move_x_mm = int(error_x * self.pixel_to_mm_ratio * 20000)
        move_y_mm = int(error_y * self.pixel_to_mm_ratio * 6335)
        self.motor_controller.move_x_to_relative(move_x_mm)
        self.motor_controller.move_y_to_relative(-move_y_mm)

        self.pupil_alignment_mode = not self.pupil_alignment_mode
        self.pupil_align_button.setText("开始瞳孔对齐")
        self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")

        # 更新状态显示
        x_status = "已对准" if self.x_aligned else "调整中"
        y_status = "已对准" if self.y_aligned else "调整中"

        self.alignment_status_text.setText(
            f"正在对齐...\n"
            f"X偏差: {error_x:.1f} 像素 - {x_status}\n"
            f"Y偏差: {error_y:.1f} 像素 - {y_status}\n"
            f"活动线程数: {len([t for t in self.motor_threads if t.is_alive()])}"
        )

    def start_mouse_control(self, target_x, target_y):
        """开始鼠标控制 - 发送一次移动指令"""
        if self.camera is None or self.motor_controller is None:
            return

        # 控制参数
        speed_factor = 1.5  # 速度系数，可以调整
        dead_zone = 50  # 死区大小（像素）
        max_speed = 800  # 最大速度

        # 计算相对于图像中心的偏差
        image_center_x = self.image_label.size().width() / 2
        image_center_y = self.image_label.size().height() / 2

        error_x = target_x - image_center_x
        error_y = target_y - image_center_y

        # 计算速度（比例控制）
        speed_x = error_x * speed_factor
        speed_y = error_y * speed_factor

        # 限制最大速度
        speed_x = max(-max_speed, min(max_speed, speed_x))
        speed_y = max(-max_speed, min(max_speed, speed_y))

        # 应用死区
        if abs(error_x) < dead_zone:
            speed_x = 0
        if abs(error_y) < dead_zone:
            speed_y = 0

        # 发送移动指令
        try:
            if speed_x != 0:
                self.motor_controller.move_x_go_speed(int(-speed_x))
                print(f"X轴移动速度: {int(-speed_x)}")

            if speed_y != 0:
                self.motor_controller.move_y_go_speed(int(speed_y))  # Y轴可能需要反向
                print(f"Y轴移动速度: {int(speed_y)}")

            print(f"鼠标控制开始 - 目标位置: ({target_x:.1f}, {target_y:.1f})")

        except Exception as e:
            print(f"鼠标控制启动错误: {e}")

    def stop_mouse_control(self):
        """停止鼠标控制 - 发送停止指令"""
        if self.motor_controller:
            try:
                self.motor_controller.stop_x()
                self.motor_controller.stop_y()
                print("鼠标控制停止")
            except Exception as e:
                print(f"停止电机错误: {e}")

    def update_frame(self):
        """更新相机画面 - 主线程负责获取所有图像"""
        try:
            # 只在这里获取图像
            grab_result = self.camera.RetrieveResult(100, pylon.TimeoutHandling_Return)

            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array.copy()  # 复制图像
                grab_result.Release()

                # 保存最新图像
                with self.image_lock:
                    self.latest_image = img

                # 如果检测模式开启，将图像发送给检测线程
                if self.pupil_detection_mode:
                    try:
                        # 尝试将图像放入队列（非阻塞）
                        if self.image_queue.full():
                            self.image_queue.get_nowait()  # 移除旧图像
                        self.image_queue.put_nowait(img.copy())
                    except queue.Full:
                        pass

                    # 尝试获取检测结果
                    try:
                        result = self.detection_queue.get_nowait()
                        self.process_detection_result(result)
                    except queue.Empty:
                        # 没有新结果，显示当前图像
                        self.display_current_state(img)
                else:
                    # 普通显示模式
                    self.display_normal_image(img)

                self.update_performance_info()

            elif grab_result:
                grab_result.Release()

        except Exception as e:
            print(f"更新画面错误: {e}")

    def process_detection_result(self, result: DetectionResult):
        """处理检测结果"""
        if result.pupil_circle is not None:
            x, y, radius = result.pupil_circle
            self.current_pupil_position = (x, y)

            # 绘制标记
            display_img = self.draw_pupil_markers(result.image, x, y, radius)
            self.display_cv_image(display_img)

            # 更新信息
            self.pupil_info_text.setText(
                f"瞳孔中心: ({x}, {y})\n"
                f"瞳孔半径: {radius} 像素\n"
                f"目标位置: {self.target_pupil_position}\n"
                f"检测状态: 成功"
            )

            # 如果开启了对齐模式，执行非阻塞对齐
            if self.pupil_alignment_mode:
                # def alignment():
                #     #with self.pupil_alignment_lock:
                #     try:
                #         self.perform_alignment_nonblocking(x, y)
                #     except Exception as e:
                #         print(f"对齐错误：{e}")

                t = threading.Thread(self.perform_alignment_nonblocking(x, y), daemon=True)
                t.start()
                # self.perform_alignment_nonblocking(x, y)
        else:
            self.current_pupil_position = None
            self.display_normal_image(result.image)
            self.pupil_info_text.setText("检测状态: 失败\n请调整相机参数或位置")

            if self.pupil_alignment_mode:
                self.alignment_status_text.setText("未检测到瞳孔，无法进行对齐")

    def display_current_state(self, img):
        """显示当前状态（检测进行中）"""
        if self.current_pupil_position:
            # 如果有之前的检测结果，继续显示
            x, y = self.current_pupil_position
            display_img = self.draw_pupil_markers(img, x, y, 150)  # 使用默认半径
            self.display_cv_image(display_img)
        else:
            self.display_normal_image(img)

        # 显示检测状态
        with self.detection_lock:
            if self.is_detecting:
                self.sharpness_text.append("检测中...")

    def draw_pupil_markers(self, img, x, y, radius):
        """绘制瞳孔标记"""
        if len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = img.copy()

        # 绘制瞳孔圆圈（绿色）
        cv2.circle(display_img, (x, y), radius, (0, 255, 0), 2)
        # 绘制瞳孔中心（红色）
        cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)

        # 绘制目标位置（蓝色十字）
        target_x, target_y = self.target_pupil_position
        cv2.line(display_img, (target_x - 10, target_y), (target_x + 10, target_y), (255, 0, 0), 2)
        cv2.line(display_img, (target_x, target_y - 10), (target_x, target_y + 10), (255, 0, 0), 2)

        return display_img

    def update_performance_info(self):
        """更新性能信息"""
        self.frame_count += 1

        if self.frame_count % 50 == 0:  # 每50帧更新一次
            with self.detection_lock:
                if self.detection_times:
                    avg_time = sum(self.detection_times) / len(self.detection_times)
                    self.sharpness_text.setText(
                        f"平均检测时间: {avg_time:.1f} ms\n"
                        f"最近检测时间: {self.last_detection_time:.1f} ms\n"
                        f"队列大小: {self.detection_queue.qsize()}"
                    )

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_mouse_control()
        self.close_camera()
        event.accept()

    def display_normal_image(self, img):
        self.display_cv_image(img)

    def display_cv_image(self, img):
        """将OpenCV图像转换为QPixmap并显示"""
        if len(img.shape) == 2:
            # 灰度图像
            h, w = img.shape
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # 彩色图像
            h, w, ch = img.shape
            bytes_per_line = ch * w
            if ch == 3:
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                return

        # 缩放图像以适应标签大小
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)



# ========== 主程序入口 ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PupilCameraViewer()
    viewer.show()
    sys.exit(app.exec())


