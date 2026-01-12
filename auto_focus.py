"""
自动对焦状态机模块
用于瞳孔相机的Z轴自动对焦功能
"""

import math
import time
import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import threading
from PySide6.QtCore import QObject, Signal
from scipy.optimize import curve_fit



class FocusState(Enum):
    """对焦状态枚举"""
    IDLE = "idle"
    FINDING_PUPIL = "finding_pupil"
    COARSE_FOCUSING = "coarse_focusing"
    FINE_FOCUSING = "fine_focusing"
    FOCUSED = "focused"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FocusResult:
    """对焦结果数据类"""
    success: bool
    final_position: Optional[float]
    final_sharpness: Optional[float]
    total_time: float
    message: str

def gaussian(x, A, mu, sigma):
    """一个标准的高斯函数模型"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

class AutoFocusStateMachine(QObject):
    """
    自动对焦状态机
    负责管理整个对焦流程
    """

    # Qt信号定义
    state_changed = Signal(str)  # 状态变化信号
    progress_updated = Signal(int)  # 进度更新信号(0-100)
    message_updated = Signal(str)  # 消息更新信号
    focus_completed = Signal(FocusResult)  # 对焦完成信号

    def __init__(self, motor_controller, detect_func, sharpness_func):
        """
        初始化自动对焦状态机

        Args:
            motor_controller: 电机控制器实例
            detect_func: 瞳孔检测函数，输入图像，返回(pupil_circle, sharpness)
            sharpness_func: 清晰度计算函数，输入图像，返回清晰度值
        """
        super().__init__()

        self.motor = motor_controller
        self.detect_pupil = detect_func
        self.compute_sharpness = sharpness_func

        # 状态机变量
        self.state = FocusState.IDLE
        self.is_running = False
        self.cancel_requested = False

        self.last_best_position = None # 上次对焦位置

        # 对焦参数（可调整）
        self.config = {
            # 瞳孔搜索参数
            'pupil_search_range': 80.0,  # mm
            'pupil_search_coarse_step': 3.0,  # mm
            'pupil_search_fine_step': 1,  # mm

            # 粗对焦参数
            'coarse_range': 20.0,  # mm
            'coarse_samples': 15,  # 采样点数
            'coarse_drop_ratio': 0.10,  # 当后一采样点清晰度相对前一点下降超过该比例且两点均检测到瞳孔时提前停止

            # 精对焦参数
            'fine_range': 1.0,  # mm
            'fine_focus_samples': 13,  # 采样点数（必须为奇数）
            'fine_focus_step': 0.65,  # mm
            'fine_initial_step': 2, # mm
            'fine_min_step': 0.1,  # mm
            'fine_max_iterations': 20,

            # 稳定性参数
            'settle_time': 0.05,  # 电机稳定时间(秒)
            'average_frames': 1,  # 清晰度计算平均帧数
            'pupil_detect_threshold': 0.6,  # 瞳孔检测成功率阈值

            # 清晰度参数
            'sharpness_noise_level': 0.05,  # 清晰度噪声水平
            'improvement_threshold': 0,  # 改善阈值


           'fine_focus_scan_range': 12.0,  # 连续扫描总范围 (mm)
            'fine_focus_sample_period': 0.05, # 采样周期 (s)，比如 50ms
            'fine_focus_settle_time': 0.02  # 起始/停止时的静置时间 (s)

        }

        # 对焦历史记录
        self.focus_history = []
        self.current_image = None

        # 步进电机换算：对外算法使用 mm，电机底层使用步数
        self.steps_per_mm = 20000  # 1 mm = 20000 steps

    # --- 单位换算与封装移动接口新增开始 ---
    def _mm_to_steps(self, mm: float) -> int:
        return int(round(mm * self.steps_per_mm))

    def _steps_to_mm(self, steps: float) -> float:
        return steps / self.steps_per_mm

    def _get_y_mm(self) -> float:
        steps = self.motor.get_position("y")  # 底层返回步数
        return self._steps_to_mm(steps)

    def _move_y_absolute_mm(self, y_mm: float, wait: bool = False):
        steps = self._mm_to_steps(y_mm)
        self.motor.move_to_absolute("y", steps, wait)

    def _move_y_relative_mm(self, delta_mm: float, wait: bool = False):
        steps = self._mm_to_steps(delta_mm)
        self.motor.move_to_relative("y", steps, wait)

    def set_config(self, **kwargs):
        """更新配置参数"""
        self.config.update(kwargs)

    def set_image_source(self, get_image_func):
        """
        设置图像获取函数
        Args:
            get_image_func: 无参数函数，返回当前相机图像
        """
        self.get_image = get_image_func

    def start_auto_focus(self):
        """
        启动自动对焦（异步）
        """
        if self.is_running:
            self.message_updated.emit("Focus already in progress...")
            return

        self.cancel_requested = False
        self.is_running = True

        # 在新线程中执行对焦
        focus_thread = threading.Thread(target=self._run_focus_sequence, daemon=True)
        focus_thread.start()

    def cancel_focus(self):
        """取消对焦"""
        self.cancel_requested = True
        self.message_updated.emit("Cancelling focus...")

    def _change_state(self, new_state: FocusState):
        """改变状态并发送信号"""
        self.state = new_state
        self.state_changed.emit(new_state.value)

    def _update_progress(self, progress: int):
        """更新进度"""
        self.progress_updated.emit(min(100, max(0, progress)))

    def _run_focus_sequence(self):
        """
        执行完整的对焦序列
        """


        self.motor._get_axis("y")._impl._drv.initPPMode()
        if self.motor._get_axis("y")._impl._drv.isFault():
            self.motor._get_axis("y")._impl._drv.faultReset()
        self.motor._get_axis("y")._impl._drv.enableOperation()

        start_time = time.time()



        try:

            if self._check_pupil_detection():
                if self.last_best_position is not None:
                    current_y = self.last_best_position
                    pupil_position = current_y
                else:
                    current_y = self._get_y_mm()
                    pupil_position = current_y
                self.message_updated.emit(f"Pupil detected at current position: {current_y:.2f}mm")
                self.message_updated.emit(f"Starting auto focus, current position: {current_y:.2f}mm")

            else:
                if self.last_best_position is not None:
                    current_y = self.last_best_position
                else:
                    # 获取当前Y轴位置(步->mm)
                    current_y = self._get_y_mm()
                self.message_updated.emit(f"Starting auto focus, current position: {current_y:.2f}mm")

                # 阶段1：搜索瞳孔
                self._change_state(FocusState.FINDING_PUPIL)
                self._update_progress(10)

                pupil_position = self._find_pupil_position(current_y)

            self._move_y_relative_mm(-6, wait=True)
            # while self.motor._get_axis("y").isRunning():
            #     time.sleep(0.01)

            if self._check_pupil_detection():
                pupil_position = pupil_position - 6
            else:
                pupil_position = pupil_position - 3

            if self.cancel_requested:
                self._handle_cancel()
                return

            if pupil_position is None:
                self.last_best_position = None
                self._handle_focus_failed("Pupil not found")
                return
            else:
                self.message_updated.emit(f"Found pupil position: {pupil_position:.2f}mm")

            # 阶段2：精对焦
            self._change_state(FocusState.FINE_FOCUSING)
            self._update_progress(50)

            # final_y, final_sharpness = self._fine_focus(pupil_position)
            # final_y, final_sharpness = self._fine_focus_climb_hill(pupil_position)
            final_y, final_sharpness = self._fine_focus_continuous_pp(pupil_position)

            if self.cancel_requested:
                self._handle_cancel()
                return

            if final_y is None:
                self.last_best_position = None
                self._handle_focus_failed("Fine focus failed")
                return

            # 对焦成功
            self._change_state(FocusState.FOCUSED)
            self._update_progress(100)

            elapsed_time = time.time() - start_time
            self.last_best_position = final_y
            result = FocusResult(
                success=True,
                final_position=final_y,
                final_sharpness=final_sharpness,
                total_time=elapsed_time,
                message=f"Focus successful! Position: {final_y:.3f}mm, Sharpness: {final_sharpness:.2f}"
            )

            self.message_updated.emit(result.message)
            self.focus_completed.emit(result)

        except Exception as e:
            self._handle_focus_failed(f"Focus exception: {str(e)}")

        finally:
            self.is_running = False

    def _find_pupil_position(self, center_y: float) -> Optional[float]:
        """
        搜索能检测到瞳孔的Y轴位置
        """
        self.message_updated.emit("Searching for pupil...")

        max_range = self.config['pupil_search_range']
        coarse_step = self.config['pupil_search_coarse_step']

        # 第一遍：粗搜索
        positions = self._generate_spiral_positions(center_y, max_range / 2, coarse_step)

        for i, y_pos in enumerate(positions):
            if self.cancel_requested:
                return None

            self._update_progress(10 + int(20 * i / len(positions)))

            # 直接绝对移动到目标 mm 位置
            self._move_y_absolute_mm(y_pos)
            if i == 0:
                while self.motor._get_axis("y").isRunning():
                    time.sleep(0.05)
            if self._check_pupil_detection():
                self.last_best_position = y_pos
                # 找到瞳孔后立即返回当前位置
                return y_pos

        # 如果没有找到瞳孔
        return None




    def _fine_focus(self, start_y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        精细对焦（基于正态曲线拟合）
        通过采集焦点附近的多个点，拟合出清晰度曲线，并直接计算出峰值位置。
        """
        self.message_updated.emit(f"Starting fine focus, start position: {start_y:.2f}mm")

        # 1. 定义采样参数：从起点往前步进9步，步长0.8mm
        num_samples = self.config['fine_focus_samples']
        step_mm = self.config['fine_focus_step']

        # 从起点开始往前生成采样点序列
        positions = np.array([start_y + i * step_mm for i in range(num_samples)])

        sampled_data = []

        # 2. 采集清晰度数据
        self.message_updated.emit(f"Sampling in range {positions[0]:.2f}mm to {positions[-1]:.2f}mm...")
        for i, y_pos in enumerate(positions):
            if self.cancel_requested:
                return None, None

            # 更新进度条 (精对焦阶段占60%到90%的进度)
            progress = 60 + int(30 * (i + 1) / num_samples)
            self._update_progress(progress)

            # 移动电机并测量清晰度
            self._move_y_absolute_mm(y_pos, wait=False)
            time.sleep(self.config['settle_time'])
            if i == 0:
                while self.motor._get_axis("y").isRunning():
                    time.sleep(0.05)



            sharpness = self._measure_sharpness_averaged(require_pupil=True)

            if sharpness is not None and sharpness > 0:
                sampled_data.append({'position': y_pos, 'sharpness': sharpness})
                self.message_updated.emit(f"Sample point {i + 1}/{num_samples}: Position={y_pos:.3f}mm, Sharpness={sharpness:.2f}")
            else:
                self.message_updated.emit(f"Sample point {i + 1}/{num_samples}: Position={y_pos:.3f}mm, No pupil detected or invalid sharpness")

        # 3. 数据校验与曲线拟合
        # 必须至少有3个有效数据点才能进行二次拟合
        if len(sampled_data) < 3:
            self.message_updated.emit("Insufficient valid sample points, cannot perform curve fitting")
            # 如果一个有效点都没有，则对焦失败
            if not sampled_data:
                return None, None
            # 如果有少量点，则退回到选择清晰度最高的那个点作为最佳位置
            best_sampled_point = max(sampled_data, key=lambda x: x['sharpness'])
            best_y = best_sampled_point['position']
            best_sharpness = best_sampled_point['sharpness']
            self.message_updated.emit(f"Fallback strategy: Moving to best sample point {best_y:.3f}mm")
            self._move_y_absolute_mm(best_y, wait=True)
            # time.sleep(self.config['settle_time'])
            # while self.motor._get_axis("y").isRunning():
            #     time.sleep(0.01)
            return best_y, best_sharpness

        # 1. 定义高斯/正态分布函数模型
        def gaussian(x, A, mu, sigma):
            """一个标准的高斯函数模型"""
            return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        # 提取数据用于拟合 (此部分与您的原代码相同)
        x_data = np.array([d['position'] for d in sampled_data])
        y_data = np.array([d['sharpness'] for d in sampled_data])

        best_y = None

        try:
            # 2. 为参数提供合理的初始猜测值
            #    振幅(A)猜测为数据中的最大值
            #    均值(mu)猜测为最大值对应的位置
            #    标准差(sigma)猜测为x数据的标准差，这是一个稳健的初始值
            initial_guess = [
                np.max(y_data),
                x_data[np.argmax(y_data)],
                np.std(x_data)
            ]

            # 3. 执行高斯曲线拟合
            #    popt 数组中将包含优化后的参数 [A, mu, sigma]
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess)

            # 提取拟合后的参数
            A_fit, mu_fit, sigma_fit = popt

            # 4. 合理性检查：
            #    - 振幅 A 和宽度 sigma 应该为正数
            #    - 预测的峰值位置 mu 应该落在采样区间内
            min_pos, max_pos = np.min(x_data), np.max(x_data)

            if A_fit > 0 and sigma_fit > 0 and min_pos <= mu_fit <= max_pos:
                # 拟合成功且结果合理，mu_fit 就是我们要找的最佳位置
                best_y = mu_fit
                self.message_updated.emit(f"Gaussian fitting successful! Theoretical optimal position: {best_y:.3f}mm")
            else:
                # 构造详细的错误信息
                reasons = []
                if A_fit <= 0:
                    reasons.append("non-positive amplitude")
                if sigma_fit <= 0:
                    reasons.append("non-positive width")
                if not (min_pos <= mu_fit <= max_pos):
                    reasons.append("predicted peak outside sampling range")
                self.message_updated.emit(f"Gaussian fitting results unreliable: {', '.join(reasons)}.")

        except RuntimeError:
            # curve_fit 在无法收敛时会抛出 RuntimeError
            self.message_updated.emit("Gaussian curve fitting failed, data may not be bell-shaped.")
        except Exception as e:
            # 捕获其他可能的未知错误
            self.message_updated.emit(f"Unknown error occurred during fitting calculation: {e}")

        # 如果拟合失败或结果不可靠，则采用回退策略：选择采样的所有点中清晰度最高的那个点
        if best_y is None:
            best_sampled_point = max(sampled_data, key=lambda x: x['sharpness'])
            best_y = best_sampled_point['position']
            self.message_updated.emit(f"Fallback strategy: Moving to best sample point {best_y:.3f}mm")

        # 4. 移动到最终计算出的最佳位置并获取最终清晰度
        self._update_progress(95)
        self._move_y_absolute_mm(best_y, wait=True)
        # 最终移动后可以稍微多等待一会，确保电机完全稳定
        # while self.motor._get_axis("y").isRunning():
        #     time.sleep(0.01)
        final_sharpness = self._measure_sharpness_averaged(require_pupil=True)
        # self.motor._get_axis("y").stop()

        # 如果在最终位置未能测得清晰度（例如，由于微小误差导致瞳孔丢失），
        # 则使用拟合曲线的理论峰值或采样的最大值作为最终清晰度
        if final_sharpness is None:
            if best_y is not None and 'peak_y' in locals() and best_y == peak_y:
                # 使用拟合函数计算理论清晰度
                final_sharpness = np.polyval(coeffs, best_y)
            else:
                # 使用采样到的最大清晰度
                final_sharpness = max(y_data)

        self.message_updated.emit(f"Fine focus completed. Final position: {best_y:.3f}mm, Sharpness: {final_sharpness:.2f}")

        return best_y, final_sharpness

    def _fine_focus_continuous_pp(self, start_y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        精细对焦（连续扫描版，位置模式 + 高斯拟合）：
        从 start_y 开始往正方向连续走一段距离（默认 12mm），
        在运动过程中不断采样 (y, sharpness)，
        扫描结束后对采样点做高斯拟合，取拟合峰值位置为最佳焦点。
        """
        self.message_updated.emit(
            f"[Continuous-PP] Starting fine focus from {start_y:.3f}mm"
        )

        scan_range = float(self.config.get('fine_focus_scan_range', 12.0))  # 总扫描范围 mm
        sample_period = float(self.config.get('fine_focus_sample_period', 0.05))  # 采样周期 s
        settle_time = float(self.config.get('fine_focus_settle_time', 0.02))  # 起止静置时间 s

        if scan_range <= 0:
            self.message_updated.emit("[Continuous-PP] scan_range must be > 0")
            return None, None

        # 扫描区间：start_y -> start_y + scan_range
        y_start = start_y
        y_end = start_y + scan_range

        # 1️⃣ 先把电机准确移动到 y_start，并静置一小会
        self._move_y_absolute_mm(y_start, wait=True)
        time.sleep(settle_time)

        axis_y = self.motor._get_axis("y")

        # 2️⃣ 发一条“移动到 y_end”的绝对位置指令（非阻塞），让电机连续走
        self._move_y_absolute_mm(y_end, wait=False)
        self.message_updated.emit(
            f"[Continuous-PP] Scanning from {y_start:.3f}mm to {y_end:.3f}mm..."
        )

        sampled_data: list[dict] = []
        last_sample_t = time.time()

        try:
            while axis_y.isRunning():
                if self.cancel_requested:
                    self.message_updated.emit("[Continuous-PP] Cancel requested, stopping.")
                    axis_y.stop()
                    return None, None

                now = time.time()
                # 控制采样频率
                if now - last_sample_t < sample_period:
                    time.sleep(0.001)
                    continue
                last_sample_t = now

                # 读当前 Y 位置（mm）
                current_y = self._get_y_mm()

                # 防守式保护：如果明显超过 y_end，就退出
                if current_y > y_end + 0.5:
                    self.message_updated.emit(
                        f"[Continuous-PP] current_y {current_y:.3f}mm > y_end {y_end:.3f}mm, stop sampling."
                    )
                    break

                # 更新进度（精对焦阶段 60%~90%）
                progress_ratio = (current_y - y_start) / max(1e-6, (y_end - y_start))
                progress = 60 + int(30 * max(0.0, min(1.0, progress_ratio)))
                self._update_progress(progress)

                # 拍照 + 算清晰度
                sharpness = self._measure_sharpness_averaged(require_pupil=True)
                if sharpness is not None and sharpness > 0:
                    sampled_data.append({
                        'position': current_y,
                        'sharpness': float(sharpness),
                    })
                    self.message_updated.emit(
                        f"[Continuous-PP] Sample: y={current_y:.3f}mm, Sharpness={sharpness:.2f}"
                    )

                time.sleep(0.001)  # 避免 CPU 忙等

        finally:
            # 确保最后电机停下来
            # axis_y.stop()
            time.sleep(settle_time)

        # 3️⃣ 扫描完成后，分析采样数据 + 高斯拟合
        if not sampled_data:
            self.message_updated.emit("[Continuous-PP] No valid samples, fine focus failed.")
            return None, None

        # 至少要有 3 个点才能拟合
        if len(sampled_data) < 3:
            self.message_updated.emit(
                f"[Continuous-PP] Only {len(sampled_data)} valid samples, fallback to max-sharpness point."
            )
            best_sample = max(sampled_data, key=lambda d: d['sharpness'])
            best_y = best_sample['position']
            best_sharpness = best_sample['sharpness']
        else:
            # 高斯模型
            def gaussian(x, A, mu, sigma):
                return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            x_data = np.array([d['position'] for d in sampled_data], dtype=float)
            y_data = np.array([d['sharpness'] for d in sampled_data], dtype=float)

            A0 = float(np.max(y_data))
            mu0 = float(x_data[np.argmax(y_data)])
            sigma0 = float(np.std(x_data))
            if sigma0 <= 0:
                sigma0 = float((np.max(x_data) - np.min(x_data)) / 4.0) or 1.0

            popt = None
            best_y = None
            best_sharpness = None

            try:
                popt, pcov = curve_fit(gaussian, x_data, y_data, p0=[A0, mu0, sigma0])
                A_fit, mu_fit, sigma_fit = popt
                min_pos, max_pos = float(np.min(x_data)), float(np.max(x_data))

                reasons = []
                if A_fit <= 0:
                    reasons.append("A <= 0")
                if sigma_fit <= 0:
                    reasons.append("sigma <= 0")
                if not (min_pos <= mu_fit <= max_pos):
                    reasons.append("mu outside sampling range")

                if not reasons:
                    # 拟合结果看起来合理
                    best_y = float(mu_fit)
                    best_sharpness = float(gaussian(best_y, *popt))
                    self.message_updated.emit(
                        f"[Continuous-PP] Gaussian fit OK: A={A_fit:.2f}, mu={best_y:.3f}, sigma={sigma_fit:.3f}"
                    )
                else:
                    # 拟合结果不靠谱，退回到最大样本点
                    self.message_updated.emit(
                        f"[Continuous-PP] Gaussian fit unreliable: {', '.join(reasons)}, "
                        "fallback to max-sharpness sample."
                    )
                    best_sample = max(sampled_data, key=lambda d: d['sharpness'])
                    best_y = best_sample['position']
                    best_sharpness = best_sample['sharpness']

            except Exception as e:
                # 拟合失败，退回最大样本点
                self.message_updated.emit(
                    f"[Continuous-PP] Gaussian fitting failed: {e}, fallback to max-sharpness sample."
                )
                best_sample = max(sampled_data, key=lambda d: d['sharpness'])
                best_y = best_sample['position']
                best_sharpness = best_sample['sharpness']

        self.message_updated.emit(
            f"[Continuous-PP] Best position (after fitting/fallback): {best_y:.3f}mm, "
            f"Sharpness≈{best_sharpness:.2f}"
        )

        # 4️⃣ 再移动回最佳位置，测一次最终清晰度
        self._update_progress(95)
        self._move_y_absolute_mm(best_y, wait=True)
        time.sleep(settle_time)
        final_sharpness = self._measure_sharpness_averaged(require_pupil=True)
        if final_sharpness is None:
            final_sharpness = best_sharpness

        self.message_updated.emit(
            f"[Continuous-PP] Fine focus done. Final position: {best_y:.3f}mm, "
            f"Sharpness: {final_sharpness:.2f}"
        )

        return best_y, final_sharpness

    def _measure_sharpness_averaged(self, require_pupil: bool = False) -> Optional[float]:
        """
        测量清晰度（多帧平均）
        """
        sharpness_values = []

        for _ in range(self.config['average_frames']):
            img = self.get_image()

            if require_pupil:
                # 需要检测到瞳孔才计算虹膜清晰度
                pupil_circle, sharpness = self.detect_pupil(img)
                if pupil_circle is not None and sharpness > 0:
                    sharpness_values.append(sharpness)
            else:
                # 使用全局清晰度
                sharpness = self.compute_sharpness(img)
                if sharpness > 0:
                    sharpness_values.append(sharpness)

        if sharpness_values:
            return np.mean(sharpness_values)
        return None

    def _check_pupil_detection(self) -> bool:
        """
        检查当前位置是否能检测到瞳孔
        """
        detected_count = 0
        check_frames = 2

        for _ in range(check_frames):
            img = self.get_image()
            pupil_circle, _ = self.detect_pupil(img)
            if pupil_circle is not None:
                detected_count += 1

        return detected_count >= 1

    def _generate_spiral_positions(self, center: float, max_range: float, step: float) -> List[float]:
        """
        生成位置列表：从中心到最大，后半段是从最小到中间。
        """
        if step <= 0 or max_range <= 0:
            return [center]

        n = int(math.floor(max_range / step))

        # 前半段：从center到最大值
        first_half = [center + k * step for k in range(0, n + 1)]

        # 后半段：从最小值到center（不包含center避免重复）
        second_half = [center + k * step for k in range(-n, 0)]

        return first_half + second_half

    def _predict_peak_position(self, results: List[Dict]) -> Optional[float]:
        """
        使用二次拟合预测峰值位置
        """
        if len(results) < 3:
            return None

        positions = np.array([r['position'] for r in results])
        sharpness = np.array([r['sharpness'] for r in results])

        # 去除零值
        valid = sharpness > 0
        if np.sum(valid) < 3:
            return None

        positions = positions[valid]
        sharpness = sharpness[valid]

        try:
            # 二次多项式拟合
            coeffs = np.polyfit(positions, sharpness, 2)

            # 检查是否为凸函数
            if coeffs[0] < 0:
                # 计算峰值位置
                peak_position = -coeffs[1] / (2 * coeffs[0])
                return peak_position
        except:
            pass

        return None

    def _handle_cancel(self):
        """处理取消操作"""
        self._change_state(FocusState.CANCELLED)
        self.message_updated.emit("Focus cancelled")

        result = FocusResult(
            success=False,
            final_position=None,
            final_sharpness=None,
            total_time=0,
            message="User cancelled focus"
        )
        self.focus_completed.emit(result)

    def _handle_focus_failed(self, reason: str):
        """处理对焦失败"""
        self._change_state(FocusState.FAILED)
        self.message_updated.emit(f"Focus failed: {reason}")

        result = FocusResult(
            success=False,
            final_position=None,
            final_sharpness=None,
            total_time=0,
            message=reason
        )
        self.focus_completed.emit(result)



    # 基于爬山法的精对焦方法
    def _fine_focus_climb_hill(self, start_y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        精细对焦（改进的爬山算法）
        """
        self.message_updated.emit(f"Starting fine focus, initial position: {start_y:.2f}mm")

        current_y = start_y
        step = self.config['fine_initial_step']
        min_step = self.config['fine_min_step']
        max_iterations = self.config['fine_max_iterations']

        best_y = current_y
        best_sharpness = 0
        no_improvement_count = 0

        k = 0

        for iteration in range(max_iterations):
            if self.cancel_requested:
                return None, None

            progress = 60 + int(40 * iteration / max_iterations)
            self._update_progress(progress)

            # 评估三个位置
            positions = [current_y - step, current_y, current_y + step]
            sharpness_values = []

            for y_pos in positions:
                self._move_y_absolute_mm(y_pos)
                time.sleep(max((self.config['settle_time'] * 1.0 * step / 0.8), 0.05))
                # while self.motor._get_axis("y").isRunning():
                #     pass

                sharpness = self._measure_sharpness_averaged(require_pupil=True)
                sharpness_values.append(sharpness if sharpness is not None else 0)

            # 找最佳位置
            max_idx = np.argmax(sharpness_values)
            max_sharpness = sharpness_values[max_idx]

            self.message_updated.emit(
                f"Iteration {iteration + 1}: Step={step:.3f}mm, "
                f"Sharpness=[{sharpness_values[0]:.2f}, {sharpness_values[1]:.2f}, {sharpness_values[2]:.2f}]"
            )

            # 更新最佳记录
            if max_sharpness > best_sharpness * (1 + self.config['improvement_threshold']):
                best_sharpness = max_sharpness
                best_y = positions[max_idx]
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # 判断收敛
            if max_idx == 1:  # 中心点最佳
                step *= 0.5
                if step < min_step:
                    self.message_updated.emit("Minimum step size reached, focus completed")
                    break
            else:
                current_y = positions[max_idx]

            # 早停条件
            if no_improvement_count >= 3:
                self.message_updated.emit("No significant improvement in sharpness, focus completed")
                break

        # 移动到最佳位置
        self._move_y_absolute_mm(best_y, wait=True)
        # while self.motor._get_axis("y").isRunning():
        #     time.sleep(0.05)

        best_sharpness = self._measure_sharpness_averaged(require_pupil=True)
        return best_y, best_sharpness


    def _coarse_focus(self, y_min: float, y_max: float) -> Optional[float]:
        """
        粗对焦搜索（带增强的提前停止逻辑）。
        在扫描过程中动态检测清晰度峰值，一旦发现峰值模式立即停止并进行拟合。
        """
        self.message_updated.emit(f"Coarse focus search: {y_min:.2f}mm to {y_max:.2f}mm")

        n_samples = self.config['coarse_samples']
        positions = np.linspace(y_min, y_max, n_samples)

        all_results = []
        # 使用一个列表来维护最近的3个有效（检测到瞳孔且清晰度大于0）结果
        recent_valid_results = []

        predicted_peak_pos = None

        for i, y_pos in enumerate(positions):
            if self.cancel_requested:
                return None
            self._update_progress(30 + int(30 * (i + 1) / len(positions)))

            self._move_y_absolute_mm(y_pos)
            time.sleep(self.config['settle_time'])
            # while self.motor._get_axis("y").isRunning():
            #     time.sleep(0.01)

            has_pupil = self._check_pupil_detection()
            if has_pupil:
                sharpness = self._measure_sharpness_averaged(require_pupil=True)
            else:
                sharpness = self._measure_sharpness_averaged()

            current_result = {
                'position': y_pos,
                'sharpness': sharpness if sharpness is not None else 0,
                'has_pupil': has_pupil
            }
            all_results.append(current_result)
            self.message_updated.emit(f"Position {y_pos:.2f}mm, Sharpness: {current_result['sharpness']:.2f}")

            # 只有当检测到瞳孔且清晰度有效时，才将其加入用于峰值判断的列表
            if current_result['has_pupil'] and current_result['sharpness'] > 0:
                recent_valid_results.append(current_result)
                # 保持列表长度不超过3
                if len(recent_valid_results) > 3:
                    recent_valid_results.pop(0)

            # --- 增强的提前停止逻辑 ---
            if len(recent_valid_results) == 3:
                p1, p2, p3 = recent_valid_results
                # 检查是否形成 S1 < S2 > S3 的峰值模式
                if p1['sharpness'] < p2['sharpness'] and p2['sharpness'] > p3['sharpness']:
                    self.message_updated.emit(f"Detected sharpness peak pattern, stopping coarse focus early.")

                    # 使用这三点进行二次拟合来预测精确峰值
                    predicted_peak_pos = self._predict_peak_position(recent_valid_results)

                    # 合理性检查：确保预测值在形成峰值的三个点之间
                    if predicted_peak_pos and (p1['position'] <= predicted_peak_pos <= p3['position']):
                        self.message_updated.emit(f"Fitting predicts peak at: {predicted_peak_pos:.3f}mm")
                        # 成功找到峰值，跳出循环
                        break
                    else:
                        # 如果拟合失败或结果不可靠，则回退到采用三个点中清晰度最高的那个点
                        self.message_updated.emit(f"Fitting failed or unreliable, using peak sample point {p2['position']:.3f}mm")
                        predicted_peak_pos = p2['position']
                        break

        # --- 循环结束后处理结果 ---

        # 如果通过提前停止逻辑成功找到了峰值，直接返回结果
        if predicted_peak_pos is not None:
            return predicted_peak_pos

        # --- 回退策略：如果循环正常结束（未提前停止），则分析所有采集到的数据 ---
        self.message_updated.emit("Full range scan completed, analyzing optimal position...")

        # 优先选择那些检测到瞳孔的结果进行分析
        pupil_results = [r for r in all_results if r['has_pupil'] and r['sharpness'] > 0]

        # 如果没有任何检测到瞳孔的结果，则放宽条件，使用所有结果
        if not pupil_results:
            if not all_results: return None  # 没有任何数据
            pupil_results = all_results

        # 尝试使用所有有效的采样点进行一次全局拟合
        if len(pupil_results) >= 3:
            predicted_peak = self._predict_peak_position(pupil_results)
            # 检查预测值是否在扫描范围内
            if predicted_peak and y_min <= predicted_peak <= y_max:
                self.message_updated.emit(f"Global fitting predicts peak at: {predicted_peak:.3f}mm")
                return predicted_peak

        # 如果拟合失败或数据点不足，则返回单个清晰度最高的采样点位置
        best_point = max(pupil_results, key=lambda x: x['sharpness'])
        self.message_updated.emit(f"Fallback strategy: Using best sample point {best_point['position']:.3f}mm")
        return best_point['position']