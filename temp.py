import sys
import os
import cv2
import numpy as np
from numba import njit
import queue
from dataclasses import dataclass
import time, threading
import concurrent.futures
from typing import Tuple, Optional
from Motor_Driver import Stage_LinearMovement
from dm2c import DM2C, ModbusRTU
from auto_focus import AutoFocusStateMachine, FocusState, FocusResult
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
    """
    一个简化的多轴电机控制器。
    通过将轴的配置和方法通用化，减少了代码重复。
    """

    # 1. 将配置信息提取到类级别的字典中，更清晰且易于管理
    AXIS_CONFIG = {
        'x': {
            'driver': DM2C.Driver_01,
            'high_speed': 500,
            'low_speed': 100,
            'path_speed': 800,
            'path_accel': 80,
            'path_decel': 80,
        },
        'y': {
            'driver': DM2C.Driver_02,
            'high_speed': 500,
            'low_speed': 100,
            'path_speed': 800,  # 假设与X轴相同，如果不同请修改
            'path_accel': 80,
            'path_decel': 80,
        },
        'z': {
            'driver': DM2C.Driver_03,
            'high_speed': 500,
            'low_speed': 100,
            'path_speed': 800,  # 假设与X轴相同，如果不同请修改
            'path_accel': 80,
            'path_decel': 80,
        },
        # 如果未来有R轴或其他轴，只需在此处添加即可
        # 'r': { ... }
    }

    # 其他设备级别的配置
    SERIAL_PORT = "COM7"
    BAUDRATE = 115200

    def __init__(self):
        """初始化Modbus连接和所有在配置中定义的轴。"""
        self.modbus = ModbusRTU(
            serial.Serial(port=self.SERIAL_PORT, baudrate=self.BAUDRATE, timeout=2, write_timeout=2))
        self.axes = {}

        # 2. 使用循环来初始化每个轴，避免代码重复
        for axis_name, config in self.AXIS_CONFIG.items():
            axis_controller = Stage_LinearMovement(config['driver'])
            axis_controller.setModbus(self.modbus)
            axis_controller.reset()
            # 根据配置设置路径和速度
            axis_controller.setRelativePositionPath(speed=config['path_speed'], acceleration=config['path_accel'],
                                                    deceleration=config['path_decel'])
            axis_controller.setAbsolutePositionPath()
            axis_controller.setSpeedPath()
            axis_controller.setJogSpeed(config['low_speed'], config['high_speed'])

            self.axes[axis_name] = axis_controller

    def _get_axis(self, axis_name: str):
        """一个辅助方法，用于获取指定轴的控制器实例并处理错误。"""
        axis_controller = self.axes.get(axis_name.lower())
        if not axis_controller:
            raise ValueError(f"无效的轴: '{axis_name}'. 可用轴: {list(self.axes.keys())}")
        return axis_controller

    # 3. 合并重复的方法
    def get_position(self, axis_name: str) -> float:
        """获取指定轴电机的当前位置（单位：steps）。"""
        return self._get_axis(axis_name).Position()

    def move_to_absolute(self, axis_name: str, position: float):
        """移动指定轴电机到绝对位置（单位：steps）。"""
        print(f"Moving {axis_name.upper()} motor to absolute position: {position:.3f} steps")
        self._get_axis(axis_name).goAbsolutePosition(position)

    def move_to_relative(self, axis_name: str, distance: float):
        """相对移动指定轴电机一定距离（单位：steps）。"""
        print(f"Moving {axis_name.upper()} motor by relative distance: {distance:.3f} steps")
        self._get_axis(axis_name).goRelativePosition(distance)

    def go_speed(self, axis_name: str, speed: float):
        """驱动指定轴电机按指定速度移动。"""
        print(f"Moving {axis_name.upper()} motor at speed: {speed}")
        self._get_axis(axis_name).goSpeed(speed)

    def stop(self, axis_name: str):
        """紧急停止指定轴的电机。"""
        print(f"Stopping {axis_name.upper()} motor")
        self._get_axis(axis_name).stop()

    def stop_all(self):
        """紧急停止所有轴的电机。"""
        print("Stopping all motors")
        for axis_name in self.axes:
            self.stop(axis_name)




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

    count = ((w - 3) // 2) * ((h - 3) // 2)

    return F * 0.1 / count


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

def extract_lower_iris_roi(img, detected_circle):
    """
    方案二：提取瞳孔下方的虹膜区域，完全避开瞳孔和上方睫毛
    """
    if detected_circle is None:
        return None, None

    x, y, radius = detected_circle
    h, w = img.shape[:2]

    print(f"瞳孔半径：{radius}像素")
    print(f"瞳孔中心位置:（{x}, {y}）")

    # 虹膜区域通常是瞳孔半径的2-3倍
    # 我们提取瞳孔下方的一个矩形虹膜区域

    # 横向：瞳孔左右各扩展1.5倍半径（虹膜宽度）
    # 纵向：从瞳孔边缘下方开始，高度为0.8倍半径
    iris_width_factor = 1.25
    roi_height_factor = 0.15

    # 计算ROI边界
    x1 = max(0, int(x - radius * iris_width_factor))
    x2 = min(w, int(x + radius * iris_width_factor))
    y1 = max(0, int(y + radius * 1.01))  # 从瞳孔边缘稍下方开始
    # y2 = min(h, int(y + radius * (1.05 + roi_height_factor)))
    y2 =  min(h, int(y + radius * 1.01 + 60))

    # 确保ROI有效
    if y2 <= y1 or x2 <= x1:
        print("警告：ROI无效，使用备用方案")
        # 备用方案：使用瞳孔下方的固定大小区域
        y1 = max(0, int(y + radius))
        y2 = min(h, y1 + 100)  # 固定高度100像素
        x1 = max(0, x - 100)
        x2 = min(w, x + 100)

    # 提取ROI
    roi = img[y1:y2, x1:x2]

    # 返回ROI和位置信息
    roi_info = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'center': (x, y), 'radius': radius
    }

    return roi, roi_info

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
    _, fixed_thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)
    thresholds.append(fixed_thresh)

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

            if radius < r_threshold or radius > 1.8 * r_threshold:  # 半径范围检查
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

    # ===========取瞳孔下方虹膜区域计算清晰度========================
    roi, roi_info = extract_lower_iris_roi(img, detected_circle)

    if roi is None or roi.size == 0:
        return None, 0.0

    # 计算清晰度（使用优化后的ROI）
    sharpness = compute_image_sharpness_numba(roi)

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
        self.timer.timeout.connect(self.update_frame)
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

        # ============ 自动对焦相关 ============
        self.auto_focus_mode = False  # 自动对焦模式标志
        self.auto_focus_machine = None  # 自动对焦状态机
        self.last_focus_result = None  # 最后对焦结果

        # 创建UI
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面 - 两列布局版本"""
        main_layout = QHBoxLayout()

        # ========== 左侧控制区域（两列布局） ==========
        control_container = QWidget()
        control_container.setMaximumWidth(800)  # 增加宽度以容纳两列

        # 创建两列布局
        two_column_layout = QHBoxLayout()

        # ========== 第一列：相机参数和控制 ==========
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        # 相机参数控制
        left_column.addWidget(QLabel("=== Camera Parameters Setup ==="))

        # 曝光模式
        self.exposure_mode_label = QLabel("Exposure Mode:")
        self.exposure_mode_combo = QComboBox()
        self.exposure_mode_combo.addItems(["Timed", "TriggerWidth"])
        left_column.addWidget(self.exposure_mode_label)
        left_column.addWidget(self.exposure_mode_combo)

        # 曝光时间
        self.exposure_time_label = QLabel("Exposure Time (us):")
        self.exposure_time_edit = QLineEdit("10000")
        self.exposure_time_edit.setValidator(QIntValidator(30, 10000000))
        left_column.addWidget(self.exposure_time_label)
        left_column.addWidget(self.exposure_time_edit)

        # 自动曝光
        self.auto_exposure_label = QLabel("Auto Exposure:")
        self.auto_exposure_combo = QComboBox()
        self.auto_exposure_combo.addItems(["Off", "Once", "Continuous"])
        left_column.addWidget(self.auto_exposure_label)
        left_column.addWidget(self.auto_exposure_combo)

        # 自动增益
        self.auto_gain_label = QLabel("Auto Gain:")
        self.auto_gain_combo = QComboBox()
        self.auto_gain_combo.addItems(["Off", "Once", "Continuous"])
        left_column.addWidget(self.auto_gain_label)
        left_column.addWidget(self.auto_gain_combo)

        # 增益值
        self.gain_label = QLabel("Gain:")
        self.gain_edit = QLineEdit("0")
        self.gain_edit.setValidator(QIntValidator(0, 23))
        left_column.addWidget(self.gain_label)
        left_column.addWidget(self.gain_edit)

        # 设置参数按钮
        self.set_params_button = QPushButton("Set Parameters")
        self.set_params_button.clicked.connect(self.set_camera_parameters)
        left_column.addWidget(self.set_params_button)

        left_column.addWidget(QLabel(""))  # 空行

        # 相机控制按钮
        left_column.addWidget(QLabel("=== Camera Control ==="))

        self.open_button = QPushButton("Open Camera")
        self.open_button.clicked.connect(self.start_camera)
        left_column.addWidget(self.open_button)

        self.close_button = QPushButton("Close Camera")
        self.close_button.clicked.connect(self.close_camera)
        left_column.addWidget(self.close_button)

        self.connect_motor_button = QPushButton("Connect Motor")
        self.connect_motor_button.clicked.connect(self.connect_motor)
        left_column.addWidget(self.connect_motor_button)

        # 瞳孔检测按钮
        self.pupil_detect_button = QPushButton("Start Pupil Detection")
        self.pupil_detect_button.clicked.connect(self.toggle_pupil_detection)
        self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        left_column.addWidget(self.pupil_detect_button)

        # 瞳孔对齐按钮
        self.pupil_align_button = QPushButton("Start Pupil Alignment")
        self.pupil_align_button.clicked.connect(self.toggle_pupil_alignment)
        self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        self.pupil_align_button.setEnabled(False)  # 初始状态禁用
        left_column.addWidget(self.pupil_align_button)

        # 自动对焦控制区
        left_column.addWidget(QLabel("=== Auto Focus Control ==="))

        # Auto focus button
        self.auto_focus_button = QPushButton("Start Auto Focus")
        self.auto_focus_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.auto_focus_button.clicked.connect(self.toggle_auto_focus)
        left_column.addWidget(self.auto_focus_button)

        # Focus status label
        self.focus_state_label = QLabel("Focus Status: Idle")
        self.focus_state_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        left_column.addWidget(self.focus_state_label)

        # Focus progress bar
        self.focus_progress_bar = QProgressBar()
        self.focus_progress_bar.setRange(0, 100)
        self.focus_progress_bar.setValue(0)
        left_column.addWidget(self.focus_progress_bar)

        # Sharpness label
        self.sharpness_label = QLabel("Sharpness: --")
        left_column.addWidget(self.sharpness_label)

        # Focus info text
        self.focus_info_text = QTextEdit()
        self.focus_info_text.setMaximumHeight(100)
        self.focus_info_text.setReadOnly(True)
        left_column.addWidget(self.focus_info_text)

        # Focus parameter group (optional)
        focus_params_group = QGroupBox("Focus Parameters")
        focus_params_layout = QVBoxLayout()

        # Search range
        search_range_layout = QHBoxLayout()
        search_range_layout.addWidget(QLabel("Search Range (mm):"))
        self.search_range_spin = QDoubleSpinBox()
        self.search_range_spin.setRange(5.0, 80.0)
        self.search_range_spin.setValue(40.0)
        self.search_range_spin.setSingleStep(1.0)
        search_range_layout.addWidget(self.search_range_spin)
        focus_params_layout.addLayout(search_range_layout)

        # Precision setting
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Focus Precision (mm):"))
        self.precision_spin = QDoubleSpinBox()
        self.precision_spin.setRange(0.01, 1)
        self.precision_spin.setValue(0.1)
        self.precision_spin.setSingleStep(0.02)
        self.precision_spin.setDecimals(3)
        precision_layout.addWidget(self.precision_spin)
        focus_params_layout.addLayout(precision_layout)

        focus_params_group.setLayout(focus_params_layout)
        left_column.addWidget(focus_params_group)

        left_column.addStretch()  # 添加弹性空间

        # ========== 第二列：对齐设置和结果 ==========
        right_column = QVBoxLayout()
        right_column.setSpacing(10)

        # 对齐参数设置
        right_column.addWidget(QLabel("=== Alignment Settings ==="))

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
        right_column.addLayout(target_layout)

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

        right_column.addLayout(pid_layout)

        # 更新PID参数按钮
        self.update_pid_button = QPushButton("Update PID Parameters")
        self.update_pid_button.clicked.connect(self.update_pid_parameters)
        right_column.addWidget(self.update_pid_button)

        # y轴移动
        self.ymotor_step_label = QLabel("Step:")
        self.ymotor_step_edit = QLineEdit("10000")
        self.ymotor_step_edit.setValidator(QIntValidator(-3000000, 3000000))
        right_column.addWidget(self.ymotor_step_label)
        right_column.addWidget(self.ymotor_step_edit)
        self.ymotor_move_button = QPushButton("Move")
        self.ymotor_move_button.clicked.connect(self.ymotor_move)
        right_column.addWidget(self.ymotor_move_button)


        right_column.addWidget(QLabel(""))  # 空行

        # 检测结果显示
        right_column.addWidget(QLabel("=== Result ==="))

        # 清晰度显示
        self.sharpness_label = QLabel("Sharpness:")
        right_column.addWidget(self.sharpness_label)

        self.sharpness_text = QTextEdit()
        self.sharpness_text.setMaximumHeight(60)
        self.sharpness_text.setReadOnly(True)
        right_column.addWidget(self.sharpness_text)

        # 瞳孔信息显示
        self.pupil_info_label = QLabel("Pupil Info:")
        right_column.addWidget(self.pupil_info_label)

        self.pupil_info_text = QTextEdit()
        self.pupil_info_text.setMaximumHeight(100)
        self.pupil_info_text.setReadOnly(True)
        right_column.addWidget(self.pupil_info_text)

        # 对齐状态显示
        self.alignment_status_label = QLabel("Alignment Status:")
        right_column.addWidget(self.alignment_status_label)

        self.alignment_status_text = QTextEdit()
        self.alignment_status_text.setMaximumHeight(80)
        self.alignment_status_text.setReadOnly(True)
        right_column.addWidget(self.alignment_status_text)

        right_column.addStretch()  # 添加弹性空间

        # ========== 组合两列 ==========
        # 添加分隔线（可选）
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)

        two_column_layout.addLayout(left_column)
        two_column_layout.addWidget(separator)
        two_column_layout.addLayout(right_column)

        control_container.setLayout(two_column_layout)

        # ========== 右侧图像显示区域 ==========
        image_layout = QVBoxLayout()

        self.image_label = ClickableImageLabel(parent=self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #ccc; }")
        self.image_label.setMinimumSize(640, 480)
        image_layout.addWidget(self.image_label)

        # ========== 组合最终布局 ==========
        main_layout.addWidget(control_container)
        main_layout.addLayout(image_layout, 1)  # 图像区域占据剩余空间

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

    def ymotor_move(self):
        if self.motor_controller:
            step = int(self.ymotor_step_edit.text())
            self.motor_controller.move_to_relative("y", step)

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
                self.initialize_auto_focus()
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

    def perform_alignment_nonblocking(self, pupil_x, pupil_z):
        """非阻塞的瞳孔对齐控制"""
        try:
            target_x, target_z = self.target_pupil_position

            # 计算像素偏差
            error_x = target_x - pupil_x
            error_z = target_z - pupil_z

            # 检查对齐状态
            self.x_aligned = abs(error_x) < self.alignment_tolerance
            self.y_aligned = abs(error_z) < self.alignment_tolerance

            if self.x_aligned and self.y_aligned:
                self.alignment_status_text.setText(
                    f"对齐成功！\n"
                    f"X偏差: {error_x:.1f} 像素\n"
                    f"Y偏差: {error_z:.1f} 像素\n"
                    f"状态: 已对准"
                )
                self.pupil_alignment_mode = not self.pupil_alignment_mode
                self.pupil_align_button.setText("开始瞳孔对齐")
                self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
                self.motor_controller.stop_all()
                return

            # 清理已完成的线程
            move_x_steps = int(error_x * self.pixel_to_mm_ratio * 20000)
            move_z_steps = int(error_z * self.pixel_to_mm_ratio * 6335)
            self.motor_controller.move_to_relative("x", -move_x_steps)
            self.motor_controller.move_to_relative("z", -move_z_steps)

            self.pupil_alignment_mode = not self.pupil_alignment_mode
            self.pupil_align_button.setText("开始瞳孔对齐")
            self.pupil_align_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")

            # ==================基于PID的调整====================
            # # 清理已完成的线程
            # self.motor_threads = [t for t in self.motor_threads if t.is_alive()]
            #
            # # X轴控制（非阻塞）
            # if not self.x_aligned and len(self.motor_threads) < 2:  # 限制并发线程数
            #     control_x = self.pid_x.update(error_x)
            #     move_x_steps = int(control_x * self.pixel_to_mm_ratio * 20000)
            #     print(f"x方向控制量:{control_x} 像素")
            #
            #     def move_x():
            #         with self.motor_lock:
            #             try:
            #                 self.motor_controller.move_to_relative("x", move_x_steps)
            #             except Exception as e:
            #                 print(f"X轴移动错误: {e}")
            #
            #     t = threading.Thread(target=move_x, daemon=True)
            #     t.start()
            #     self.motor_threads.append(t)
            #     # self.motor_controller.move_to_relative("x", move_x_steps)
            #
            # # Y轴控制（非阻塞）
            # if not self.y_aligned and len(self.motor_threads) < 2:
            #     control_y = self.pid_y.update(error_z)
            #     move_z_steps = int(control_y * self.pixel_to_mm_ratio * 6300)
            #     print(f"x方向控制量:{control_y} 像素")
            #
            #     def move_y():
            #         with self.motor_lock:
            #             try:
            #                  self.motor_controller.move_to_relative("z", -move_z_steps)
            #             except Exception as e:
            #                 print(f"Y轴移动错误: {e}")
            #
            #     t = threading.Thread(target=move_y, daemon=True)
            #     t.start()
            #     self.motor_threads.append(t)
            #     self.motor_controller.move_to_relative("z", -move_z_steps)

            # 更新状态显示

            x_status = "已对准" if self.x_aligned else "调整中"
            y_status = "已对准" if self.y_aligned else "调整中"

            self.alignment_status_text.setText(
                f"正在对齐...\n"
                f"X偏差: {error_x:.1f} 像素 - {x_status}\n"
                f"Y偏差: {error_z:.1f} 像素 - {y_status}\n"
                f"活动线程数: {len([t for t in self.motor_threads if t.is_alive()])}"
            )
        except Exception as e:
            self.alignment_status_text.setText(f"对齐控制错误: {e}")

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
        error_z = target_y - image_center_y

        # 计算速度（比例控制）
        speed_x = error_x * speed_factor
        speed_z = error_z * speed_factor

        # 限制最大速度
        speed_x = max(-max_speed, min(max_speed, speed_x))
        speed_z = max(-max_speed, min(max_speed, speed_z))

        # 应用死区
        if abs(error_x) < dead_zone:
            speed_x = 0
        if abs(error_z) < dead_zone:
            speed_z = 0

        # 发送移动指令
        try:
            if speed_x != 0:
                self.motor_controller.go_speed("x", int(-speed_x))
                print(f"X轴移动速度: {int(-speed_x)}")

            if speed_z != 0:
                self.motor_controller.go_speed("z", int(speed_z))
                print(f"Z轴移动速度: {int(speed_z)}")

            print(f"鼠标控制开始 - 目标位置: ({target_x:.1f}, {target_y:.1f})")

        except Exception as e:
            print(f"鼠标控制启动错误: {e}")

    def stop_mouse_control(self):
        """停止鼠标控制 - 发送停止指令"""
        if self.motor_controller:
            try:
                self.motor_controller.stop("x")
                self.motor_controller.stop("z")
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
            # if self.pupil_alignment_mode and not self.pupil_alignment_lock.locked():
            #     t = threading.Thread(
            #         target=self.perform_alignment_nonblocking, args=(x, y), daemon=True
            #     )
            #     t.start()
            if self.pupil_alignment_mode:
                self.perform_alignment_nonblocking(x, y)
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

    def initialize_auto_focus(self):
        """初始化自动对焦状态机"""
        if self.motor_controller is None:
            self.motor_controller = MotorController()

        # 创建自动对焦状态机
        self.auto_focus_machine = AutoFocusStateMachine(
            motor_controller=self.motor_controller,
            detect_func=robust_pupil_detection,
            sharpness_func=self.compute_global_sharpness
        )

        # 设置图像获取函数
        self.auto_focus_machine.set_image_source(self.get_current_image)

        # 连接信号
        self.auto_focus_machine.state_changed.connect(self.on_focus_state_changed)
        self.auto_focus_machine.progress_updated.connect(self.on_focus_progress_updated)
        self.auto_focus_machine.message_updated.connect(self.on_focus_message_updated)
        self.auto_focus_machine.focus_completed.connect(self.on_focus_completed)

        # 更新配置
        self.update_focus_config()

    def update_focus_config(self):
        """更新对焦配置参数"""
        if self.auto_focus_machine:
            self.auto_focus_machine.set_config(
                pupil_search_range=self.search_range_spin.value(),
                fine_min_step=self.precision_spin.value()
            )

    def get_current_image(self):
        """获取当前相机图像（供自动对焦使用）"""
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.copy()

        # 如果没有缓存图像，直接从相机获取
        try:
            grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if grab_result and grab_result.GrabSucceeded():
                img = grab_result.Array
                grab_result.Release()
                return img
        except:
            pass
        return None

    def compute_global_sharpness(self, img):
        """计算全局清晰度（不依赖瞳孔检测）"""
        if img is None:
            return 0

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用图像中心40%区域
        h, w = img.shape
        center_x, center_y = w // 2, h // 2
        roi_size = int(min(w, h) * 0.4)

        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(w, center_x + roi_size // 2)
        y2 = min(h, center_y + roi_size // 2)

        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            return 0

        return compute_image_sharpness_numba(roi)

    def toggle_auto_focus(self):
        """切换自动对焦模式"""
        if not self.camera:
            QMessageBox.warning(self, "警告", "请先连接相机")
            return

        if self.auto_focus_machine is None:
            self.initialize_auto_focus()

        self.auto_focus_mode = not self.auto_focus_mode

        if self.auto_focus_mode:
            # 如果正在进行瞳孔对齐，先停止
            if self.pupil_detection_mode:
                self.toggle_pupil_detection()
            if self.pupil_alignment_mode:
                self.toggle_pupil_alignment()

            # 更新配置
            self.update_focus_config()

            # 启动自动对焦
            self.auto_focus_button.setText("停止自动对焦")
            self.auto_focus_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
            self.focus_info_text.clear()
            self.focus_info_text.append("启动自动对焦...")

            # 开始对焦
            self.auto_focus_machine.start_auto_focus()
        else:
            # 停止自动对焦
            self.auto_focus_button.setText("开始自动对焦")
            self.auto_focus_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")

            if self.auto_focus_machine.is_running:
                self.auto_focus_machine.cancel_focus()

    def on_focus_state_changed(self, state_str):
        """对焦状态变化处理"""
        self.focus_state_label.setText(f"对焦状态: {state_str}")

        # 根据状态改变标签颜色
        color_map = {
            "空闲": "#f0f0f0",
            "搜索瞳孔": "#ffeb3b",
            "粗对焦": "#ff9800",
            "精对焦": "#03a9f4",
            "对焦完成": "#4caf50",
            "对焦失败": "#f44336",
            "对焦取消": "#9e9e9e"
        }

        color = color_map.get(state_str, "#f0f0f0")
        self.focus_state_label.setStyleSheet(f"QLabel {{ background-color: {color}; padding: 5px; }}")

    def on_focus_progress_updated(self, progress):
        """对焦进度更新处理"""
        self.focus_progress_bar.setValue(progress)

    def on_focus_message_updated(self, message):
        """对焦消息更新处理"""
        self.focus_info_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # 自动滚动到底部
        scrollbar = self.focus_info_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_focus_completed(self, result: FocusResult):
        """对焦完成处理"""
        self.last_focus_result = result
        self.auto_focus_mode = False

        try:
            if getattr(result, "success", False):
                # 只有在成功对焦后再启动检测与对齐
                # 根据当前状态，避免重复切换（只在未开启时开启）
                if not self.pupil_detection_mode:
                    self.toggle_pupil_detection()
                if not self.pupil_alignment_mode:
                    self.toggle_pupil_alignment()
            else:
                # 失败时的处理（如提示或重试策略）
                pass
        except Exception as e:
            # 防御性处理，避免回调异常影响UI
            print(f"on_focus_completed 处理出错: {e}")

        self.auto_focus_button.setText("开始自动对焦")
        self.auto_focus_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")

        if result.success:
            self.focus_info_text.append(f"\n{'=' * 40}")
            self.focus_info_text.append(f"对焦成功！")
            self.focus_info_text.append(f"最终位置: {result.final_position:.3f} mm")
            self.focus_info_text.append(f"清晰度值: {result.final_sharpness:.2f}")
            self.focus_info_text.append(f"耗时: {result.total_time:.2f} 秒")
            self.focus_info_text.append(f"{'=' * 40}\n")

            # 更新清晰度显示
            self.sharpness_label.setText(f"清晰度: {result.final_sharpness:.2f}")

            QMessageBox.information(self, "对焦完成", result.message)
        else:
            self.focus_info_text.append(f"\n对焦失败: {result.message}\n")
            QMessageBox.warning(self, "对焦失败", result.message)

    def closeEvent(self, event):
        """窗口关闭事件"""

        # 停止自动对焦
        if self.auto_focus_machine and self.auto_focus_machine.is_running:
            self.auto_focus_machine.cancel_focus()
            time.sleep(0.5)  # 等待对焦线程结束

        if self.motor_controller:
            self.motor_controller.stop_all()
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


