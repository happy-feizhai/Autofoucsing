import sys
import os
import cv2
import numpy as np
from numba import njit
from typing import Tuple, Optional

# PySide6 相关导入
from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtGui import QImage, QPixmap, QIntValidator
from PySide6.QtCore import QTimer, Qt

# Basler相机相关导入
from pypylon import pylon

# 设置Qt平台插件路径
if hasattr(sys, '_MEIPASS'):
    plugin_path = os.path.join(sys._MEIPASS, 'PySide6', 'plugins', 'platforms')
else:
    plugin_path = os.path.join(os.path.dirname(QtCore.__file__), "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


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


def evaluate_contour_quality(contour: np.ndarray, center: Tuple[int, int], radius: int) -> float:
    """评估轮廓质量，返回得分"""
    # 1. 圆形度评估
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # 2. 紧凑性评估
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0

    solidity = area / hull_area

    # 3. 面积与半径一致性
    expected_area = np.pi * radius * radius
    area_ratio = min(area / expected_area, expected_area / area)

    # 综合得分
    score = circularity * 0.4 + solidity * 0.3 + area_ratio * 0.3

    return score


def evaluate_circle_quality(img: np.ndarray, circle: Tuple[int, int, int]) -> float:
    """评估圆的质量得分（基于灰度对比度）"""
    x, y, r = circle
    h, w = img.shape

    # 边界检查
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return 0.0

    # 创建mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    # 计算圆内平均灰度
    inner_mean = cv2.mean(img, mask)[0]

    # 创建环形区域
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(outer_mask, (x, y), min(int(r * 1.8), min(w, h) // 2), 255, -1)
    cv2.circle(outer_mask, (x, y), r, 0, -1)

    if cv2.countNonZero(outer_mask) == 0:
        return 0.0

    outer_mean = cv2.mean(img, outer_mask)[0]

    # 对比度得分（瞳孔应该比周围区域更暗）
    contrast = outer_mean - inner_mean

    return max(0, contrast)


def detect_pupil_contour(img: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """使用轮廓检测瞳孔"""
    # 预处理
    # processed = preprocess_image(img)

    # 多种阈值方法
    thresholds = []

    # 固定阈值（多个值）
    for thresh_val in [30, 40, 50, 60]:
        _, fixed_thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)
        thresholds.append(fixed_thresh)

    best_circle = None
    best_score = 0

    for thresh in thresholds:
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 轮廓质量评估
            area = cv2.contourArea(contour)
            if area < 2500:  # 太小的轮廓跳过
                continue

            # 拟合圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)

            if radius < 150 or radius > 250:  # 半径范围检查
                continue

            # 计算轮廓质量得分
            contour_score = evaluate_contour_quality(contour, (x, y), radius)

            # 计算灰度对比度得分
            contrast_score = evaluate_circle_quality(processed, (x, y, radius))

            # 综合得分
            total_score = contour_score * 0.6 + contrast_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_circle = (x, y, radius)

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
    sharpness = compute_image_sharpness_numba(crop)
    # sharpness = 0

    return detected_circle, sharpness


# ========== 主界面类 ==========
class PupilCameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("瞳孔识别相机系统 - Pupil Detection Camera System")
        self.setGeometry(100, 100, 1000, 800)

        # 相机相关变量
        self.camera = None
        self.timer = QTimer(self)
        self.pupil_detection_mode = False  # 瞳孔检测模式标志

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

        # 瞳孔检测按钮
        self.pupil_detect_button = QPushButton("Start Pupil Detection")
        self.pupil_detect_button.clicked.connect(self.toggle_pupil_detection)
        self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        control_panel.addWidget(self.pupil_detect_button)

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

        control_panel.addStretch()  # 添加弹性空间

        # 右侧图像显示区域
        image_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px solid #ccc; }")
        self.image_label.setMinimumSize(640, 480)
        image_layout.addWidget(self.image_label)

        # 组合布局
        control_widget = QWidget()
        control_widget.setMaximumWidth(300)
        control_widget.setLayout(control_panel)

        main_layout.addWidget(control_widget)
        main_layout.addLayout(image_layout, 1)

        self.setLayout(main_layout)

    def start_camera(self):
        """打开相机"""
        try:
            if self.camera is not None:
                self.camera.StopGrabbing()
                self.camera.Close()

            # 创建相机实例
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            # 开始捕获
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # 启动定时器更新画面
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30ms更新一次

            self.sharpness_text.setText("相机已打开")
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开相机: {str(e)}")

    def close_camera(self):
        """关闭相机"""
        if self.camera is not None:
            self.timer.stop()
            self.camera.StopGrabbing()
            self.camera.Close()
            self.camera = None

            self.sharpness_text.setText("相机已关闭")
            self.open_button.setEnabled(True)
            self.close_button.setEnabled(False)
            self.pupil_detection_mode = False
            self.pupil_detect_button.setText("开始瞳孔检测")
            self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")

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

    def toggle_pupil_detection(self):
        """切换瞳孔检测模式"""
        if self.camera is None:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return

        self.pupil_detection_mode = not self.pupil_detection_mode

        if self.pupil_detection_mode:
            self.pupil_detect_button.setText("停止瞳孔检测")
            self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
            self.sharpness_text.setText("瞳孔检测模式已开启")
        else:
            self.pupil_detect_button.setText("开始瞳孔检测")
            self.pupil_detect_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
            self.sharpness_text.setText("瞳孔检测模式已关闭")
            self.pupil_info_text.clear()

    def update_frame(self):
        """更新相机画面"""
        try:
            # 获取图像
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result is not None and grab_result.GrabSucceeded():
                img = grab_result.Array

                if self.pupil_detection_mode:
                    # 瞳孔检测模式
                    self.process_pupil_detection(img)
                else:
                    # 普通显示模式
                    self.display_normal_image(img)

                grab_result.Release()

        except Exception as e:
            print(f"更新画面错误: {str(e)}")

    def process_pupil_detection(self, img):
        """处理瞳孔检测"""
        # 转换为灰度图像进行检测
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()

        # 执行瞳孔检测
        pupil_circle, sharpness = robust_pupil_detection(gray_img)

        if pupil_circle is not None:
            # 检测到瞳孔
            x, y, radius = pupil_circle

            # 在彩色图像上绘制瞳孔标记
            if len(img.shape) == 2:
                display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = img.copy()

            # 绘制瞳孔圆圈（绿色）
            cv2.circle(display_img, (x, y), radius, (0, 255, 0), 2)
            # 绘制瞳孔中心（红色）
            cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)

            # 更新显示
            self.display_cv_image(display_img)

            # 更新文本信息
            self.sharpness_text.setText(f"清晰度: {sharpness:.2f}")
            self.pupil_info_text.setText(
                f"瞳孔中心: ({x}, {y})\n"
                f"瞳孔半径: {radius} 像素\n"
                f"检测状态: 成功"
            )
        else:
            # 未检测到瞳孔
            self.display_normal_image(img)
            self.sharpness_text.setText("未识别到瞳孔")
            self.pupil_info_text.setText("检测状态: 失败\n请调整相机参数或位置")

    def display_normal_image(self, img):
        """显示普通图像（不带标记）"""
        if len(img.shape) == 2:
            # 灰度图像转RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.close_camera()
        event.accept()


# ========== 主程序入口 ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = PupilCameraViewer()
    viewer.show()
    sys.exit(app.exec())