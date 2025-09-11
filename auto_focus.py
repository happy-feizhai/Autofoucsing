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
    IDLE = "空闲"
    FINDING_PUPIL = "搜索瞳孔"
    COARSE_FOCUSING = "粗对焦"
    FINE_FOCUSING = "精对焦"
    FOCUSED = "对焦完成"
    FAILED = "对焦失败"
    CANCELLED = "对焦取消"


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
            'fine_samples': 9,  # 采样点数（必须为奇数）
            'fine_initial_step': 0.7,  # mm
            'fine_min_step': 0.7,  # mm
            'fine_max_iterations': 20,

            # 稳定性参数
            'settle_time': 0.1,  # 电机稳定时间(秒)
            'average_frames': 1,  # 清晰度计算平均帧数
            'pupil_detect_threshold': 0.6,  # 瞳孔检测成功率阈值

            # 清晰度参数
            'sharpness_noise_level': 0.05,  # 清晰度噪声水平
            'improvement_threshold': 0,  # 改善阈值
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

    def _move_y_absolute_mm(self, y_mm: float):
        steps = self._mm_to_steps(y_mm)
        self.motor.move_to_absolute("y", steps)

    def _move_y_relative_mm(self, delta_mm: float):
        steps = self._mm_to_steps(delta_mm)
        self.motor.move_to_relative("y", steps)

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
            self.message_updated.emit("对焦正在进行中...")
            return

        self.cancel_requested = False
        self.is_running = True

        # 在新线程中执行对焦
        focus_thread = threading.Thread(target=self._run_focus_sequence, daemon=True)
        focus_thread.start()

    def cancel_focus(self):
        """取消对焦"""
        self.cancel_requested = True
        self.message_updated.emit("正在取消对焦...")

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
        start_time = time.time()


        try:
            if self.last_best_position is not None:
                current_y = self.last_best_position
            else:
                # 获取当前Y轴位置(步->mm)
                current_y = self._get_y_mm()
            self.message_updated.emit(f"开始自动对焦，当前位置: {current_y:.2f}mm")

            # 阶段1：搜索瞳孔
            self._change_state(FocusState.FINDING_PUPIL)
            self._update_progress(10)

            y_min, y_max = self._find_pupil_range(current_y)

            if self.cancel_requested:
                self._handle_cancel()
                return

            if y_min is None:
                # 未找到瞳孔，使用扩展范围
                self.message_updated.emit("未检测到瞳孔，使用扩展搜索范围")
                y_min = current_y - self.config['coarse_range']
                y_max = current_y + self.config['coarse_range']
            else:
                self.message_updated.emit(f"瞳孔检测范围: {y_min:.2f}mm - {y_max:.2f}mm")

            # 阶段2：粗对焦
            self._change_state(FocusState.COARSE_FOCUSING)
            self._update_progress(30)

            best_coarse_z = self._coarse_focus(y_min, y_max)

            if self.cancel_requested:
                self._handle_cancel()
                return

            if best_coarse_z is None:
                self.last_best_position = None
                self._handle_focus_failed("粗对焦失败")
                return

            self.message_updated.emit(f"粗对焦完成，最佳位置: {best_coarse_z:.2f}mm")

            # 阶段3：精对焦
            self._change_state(FocusState.FINE_FOCUSING)
            self._update_progress(60)

            final_y, final_sharpness = self._fine_focus(best_coarse_z)
            self._move_y_absolute_mm(final_y)

            if self.cancel_requested:
                self._handle_cancel()
                return

            if final_y is None:
                self.last_best_position = None
                self._handle_focus_failed("精对焦失败")
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
                message=f"对焦成功！位置: {final_y:.3f}mm, 清晰度: {final_sharpness:.2f}"
            )

            self.message_updated.emit(result.message)
            self.focus_completed.emit(result)

        except Exception as e:
            self._handle_focus_failed(f"对焦异常: {str(e)}")

        finally:
            self.is_running = False

    def _find_pupil_range(self, center_y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        搜索能检测到瞳孔的Y轴范围
        """
        self.message_updated.emit("正在搜索瞳孔...")

        max_range = self.config['pupil_search_range']
        coarse_step = self.config['pupil_search_coarse_step']
        fine_step = self.config['pupil_search_fine_step']

        pupil_positions = []

        # 第一遍：粗搜索
        positions = self._generate_spiral_positions(center_y, max_range / 2, coarse_step)

        for i, y_pos in enumerate(positions):
            if self.cancel_requested:
                return None, None

            self._update_progress(10 + int(20 * i / len(positions)))

            # 直接绝对移动到目标 mm 位置
            self._move_y_absolute_mm(y_pos)
            time.sleep(self.config['settle_time'])
            if i == 0:
                while self.motor._get_axis("y").isRunning():
                    time.sleep(0.1)

            if self._check_pupil_detection():
                pupil_positions.append(y_pos)

                if i == 0:
                    # 中心点就检测到瞳孔，直接返回一个小范围
                    return y_pos - 20, y_pos + 20

                # 找到瞳孔后，在附近细化搜索边界
                if len(pupil_positions) == 1:
                    return y_pos - 20, y_pos + 20
                    # # 找到第一个瞳孔位置，向两边扩展搜索
                    # for dy in np.arange(fine_step, 20.0, fine_step):
                    #     # 向正方向
                    #     y_test = y_pos + dy
                    #     self._move_y_absolute_mm(y_test)
                    #     time.sleep(self.config['settle_time'])
                    #     if self._check_pupil_detection():
                    #         pupil_positions.append(y_test)
                    #     else:
                    #         break
                    #
                    #     # 向负方向
                    #     y_test = y_pos - dy
                    #     self._move_y_absolute_mm(y_test)
                    #     time.sleep(self.config['settle_time'])
                    #     if self._check_pupil_detection():
                    #         pupil_positions.append(y_test)
                    #     else:
                    #         break
                    # break

        if pupil_positions:
            return min(pupil_positions) - 0.5, max(pupil_positions) + 0.5
        else:
            return None, None

    def _coarse_focus(self, y_min: float, y_max: float) -> Optional[float]:
        """
        粗对焦搜索（带增强的提前停止逻辑）。
        在扫描过程中动态检测清晰度峰值，一旦发现峰值模式立即停止并进行拟合。
        """
        self.message_updated.emit(f"粗对焦搜索: {y_min:.2f}mm 到 {y_max:.2f}mm")

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
            if i == 0:
                while self.motor._get_axis("y").isRunning():
                    time.sleep(0.1)

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
            self.message_updated.emit(f"位置 {y_pos:.2f}mm, 清晰度: {current_result['sharpness']:.2f}")

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
                    self.message_updated.emit(f"检测到清晰度峰值模式，提前停止粗对焦。")

                    # 使用这三点进行二次拟合来预测精确峰值
                    predicted_peak_pos = self._predict_peak_position(recent_valid_results)

                    # 合理性检查：确保预测值在形成峰值的三个点之间
                    if predicted_peak_pos and (p1['position'] <= predicted_peak_pos <= p3['position']):
                        self.message_updated.emit(f"拟合预测峰值位于: {predicted_peak_pos:.3f}mm")
                        # 成功找到峰值，跳出循环
                        break
                    else:
                        # 如果拟合失败或结果不可靠，则回退到采用三个点中清晰度最高的那个点
                        self.message_updated.emit(f"拟合失败或结果不可靠，采用峰值采样点 {p2['position']:.3f}mm")
                        predicted_peak_pos = p2['position']
                        break

        # --- 循环结束后处理结果 ---

        # 如果通过提前停止逻辑成功找到了峰值，直接返回结果
        if predicted_peak_pos is not None:
            return predicted_peak_pos

        # --- 回退策略：如果循环正常结束（未提前停止），则分析所有采集到的数据 ---
        self.message_updated.emit("完成全范围扫描，正在分析最佳位置...")

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
                self.message_updated.emit(f"全局拟合预测峰值位于: {predicted_peak:.3f}mm")
                return predicted_peak

        # 如果拟合失败或数据点不足，则返回单个清晰度最高的采样点位置
        best_point = max(pupil_results, key=lambda x: x['sharpness'])
        self.message_updated.emit(f"回退策略: 采用最佳采样点 {best_point['position']:.3f}mm")
        return best_point['position']


    def _fine_focus(self, start_y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        精细对焦（基于正态曲线拟合）
        通过采集焦点附近的多个点，拟合出清晰度曲线，并直接计算出峰值位置。
        """
        self.message_updated.emit(f"开始精对焦，中心位置: {start_y:.2f}mm")

        # 1. 定义采样参数
        num_samples = self.config['fine_samples']
        step_mm = self.config['fine_initial_step']
        half_range = (num_samples - 1) / 2 * step_mm

        # 从后往前生成采样点序列
        positions = np.linspace(start_y - half_range, start_y + half_range, num_samples)

        sampled_data = []

        # 2. 采集清晰度数据
        self.message_updated.emit(f"正在 {positions[0]:.2f}mm 到 {positions[-1]:.2f}mm 范围内采样...")
        for i, y_pos in enumerate(positions):
            if self.cancel_requested:
                return None, None

            # 更新进度条 (精对焦阶段占60%到90%的进度)
            progress = 60 + int(30 * (i + 1) / num_samples)
            self._update_progress(progress)

            # 移动电机并测量清晰度
            self._move_y_absolute_mm(y_pos)
            time.sleep(self.config['settle_time'])
            if i == 0:
                while self.motor._get_axis("y").isRunning():
                    time.sleep(0.1)


            sharpness = self._measure_sharpness_averaged(require_pupil=True)

            if sharpness is not None and sharpness > 0:
                sampled_data.append({'position': y_pos, 'sharpness': sharpness})
                self.message_updated.emit(f"采样点 {i + 1}/{num_samples}: 位置={y_pos:.3f}mm, 清晰度={sharpness:.2f}")
            else:
                self.message_updated.emit(f"采样点 {i + 1}/{num_samples}: 位置={y_pos:.3f}mm, 未检测到瞳孔或清晰度无效")

        # 3. 数据校验与曲线拟合
        # 必须至少有3个有效数据点才能进行二次拟合
        if len(sampled_data) < 3:
            self.message_updated.emit("有效采样点不足，无法进行曲线拟合")
            # 如果一个有效点都没有，则对焦失败
            if not sampled_data:
                return None, None
            # 如果有少量点，则退回到选择清晰度最高的那个点作为最佳位置
            best_sampled_point = max(sampled_data, key=lambda x: x['sharpness'])
            best_y = best_sampled_point['position']
            best_sharpness = best_sampled_point['sharpness']
            self.message_updated.emit(f"回退策略：移动到最佳采样点 {best_y:.3f}mm")
            self._move_y_absolute_mm(best_y)
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
                self.message_updated.emit(f"高斯拟合成功！理论最佳位置: {best_y:.3f}mm")
            else:
                # 构造详细的错误信息
                reasons = []
                if A_fit <= 0:
                    reasons.append("振幅非正")
                if sigma_fit <= 0:
                    reasons.append("宽度非正")
                if not (min_pos <= mu_fit <= max_pos):
                    reasons.append("预测峰值超出采样范围")
                self.message_updated.emit(f"高斯拟合结果不可靠: {', '.join(reasons)}。")

        except RuntimeError:
            # curve_fit 在无法收敛时会抛出 RuntimeError
            self.message_updated.emit("高斯曲线拟合失败，数据可能不呈钟形。")
        except Exception as e:
            # 捕获其他可能的未知错误
            self.message_updated.emit(f"拟合计算时发生未知错误: {e}")

        # 如果拟合失败或结果不可靠，则采用回退策略：选择采样的所有点中清晰度最高的那个点
        if best_y is None:
            best_sampled_point = max(sampled_data, key=lambda x: x['sharpness'])
            best_y = best_sampled_point['position']
            self.message_updated.emit(f"回退策略：移动到最佳采样点 {best_y:.3f}mm")

        # 4. 移动到最终计算出的最佳位置并获取最终清晰度
        self._update_progress(95)
        self._move_y_absolute_mm(best_y)
        # 最终移动后可以稍微多等待一会，确保电机完全稳定
        while self.motor._get_axis("y").isRunning():
            time.sleep(0.1)

        final_sharpness = self._measure_sharpness_averaged(require_pupil=True)

        # 如果在最终位置未能测得清晰度（例如，由于微小误差导致瞳孔丢失），
        # 则使用拟合曲线的理论峰值或采样的最大值作为最终清晰度
        if final_sharpness is None:
            if best_y is not None and 'peak_y' in locals() and best_y == peak_y:
                # 使用拟合函数计算理论清晰度
                final_sharpness = np.polyval(coeffs, best_y)
            else:
                # 使用采样到的最大清晰度
                final_sharpness = max(y_data)

        self.message_updated.emit(f"精对焦完成。最终位置: {best_y:.3f}mm, 清晰度: {final_sharpness:.2f}")

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
        生成位置列表：第一个为 center，其余位置按从小到大排列（不再左右交替）。
        """
        if step <= 0 or max_range <= 0:
            return [center]

        n = int(math.floor(max_range / step))
        # 升序生成：center - n*step ... center ... center + n*step
        seq = [center + k * step for k in range(-n, n + 1)]
        # 移除 center（考虑浮点误差），其余保持升序
        rest = [p for p in seq if not math.isclose(p, center, rel_tol=1e-12, abs_tol=1e-12)]
        return [center] + rest

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
        self.message_updated.emit("对焦已取消")

        result = FocusResult(
            success=False,
            final_position=None,
            final_sharpness=None,
            total_time=0,
            message="用户取消对焦"
        )
        self.focus_completed.emit(result)

    def _handle_focus_failed(self, reason: str):
        """处理对焦失败"""
        self._change_state(FocusState.FAILED)
        self.message_updated.emit(f"对焦失败: {reason}")

        result = FocusResult(
            success=False,
            final_position=None,
            final_sharpness=None,
            total_time=0,
            message=reason
        )
        self.focus_completed.emit(result)

    # 等待函数，确保电机稳定，可以替换sleep函数
    def _wait_settle(self, drop_frames: int = 3, var_threshold: float = 20):
        """
        等待运动稳定：丢弃前几帧，或者直到清晰度方差稳定。
        """
        sharpness_values = []
        for _ in range(drop_frames + 5):  # 最多检查若干帧
            frame = self.get_image()
            if frame is None:
                continue
            s = self.compute_sharpness(frame)
            sharpness_values.append(s)
            if len(sharpness_values) >= drop_frames:
                recent = sharpness_values[-drop_frames:]
                if np.var(recent) < var_threshold:
                    break
        return

    # 依据清晰度曲线的单峰性，基于黄金分割的精对焦方法，可以测试效果
    def _fine_focus_test(self, y_start: float, y_end: float, tol: float = 0.4, max_iter: int = 15) -> Tuple [Optional[float], Optional[float]]:
        """
        精细对焦（基于黄金分割搜索）
        - 在区间 [y_start, y_end] 内使用黄金分割搜索寻找清晰度最大的位置
        - 包含异常处理与回退逻辑
        """
        self.message_updated.emit(f"开始精对焦，范围: {y_start:.2f}mm ~ {y_end:.2f}mm")

        phi = (1 + 5 ** 0.5) / 2
        sampled_data = []

        # 初始化两个点
        c = y_end - (y_end - y_start) / phi
        d = y_start + (y_end - y_start) / phi

        # 移动到 c
        self._move_y_absolute_mm(c)
        self._wait_settle()
        fc = self._measure_sharpness_averaged(require_pupil=True)
        if fc is not None and fc > 0:
            sampled_data.append({'position': c, 'sharpness': fc})

        # 移动到 d
        self._move_y_absolute_mm(d)
        self._wait_settle()
        fd = self._measure_sharpness_averaged(require_pupil=True)
        if fd is not None and fd > 0:
            sampled_data.append({'position': d, 'sharpness': fd})

        # 主循环
        for i in range(max_iter):
            if self.cancel_requested:
                return None, None

            if abs(y_end - y_start) < tol:
                break

            # 更新进度条 (精对焦阶段占 60%~90%)
            progress = 60 + int(30 * (i + 1) / max_iter)
            self._update_progress(progress)

            if fc is None or fd is None:
                self.message_updated.emit("检测失败，清晰度无效，提前结束精对焦")
                break

            if fc > fd:
                # 峰值在 [y_start, d]
                y_end, d, fd = d, c, fc
                c = y_end - (y_end - y_start) / phi
                self._move_y_absolute_mm(c)
                self._wait_settle()
                fc = self._measure_sharpness_averaged()
                if fc is not None and fc > 0:
                    sampled_data.append({'position': c, 'sharpness': fc})
            else:
                # 峰值在 [c, y_end]
                y_start, c, fc = c, d, fd
                d = y_start + (y_end - y_start) / phi
                self._move_y_absolute_mm(d)
                self._wait_settle()
                fd = self._measure_sharpness_averaged()
                if fd is not None and fd > 0:
                    sampled_data.append({'position': d, 'sharpness': fd})

        # 判断是否有有效数据
        if not sampled_data:
            self.message_updated.emit("精对焦失败：没有有效的清晰度数据")
            return None, None

        # 选出采样点中清晰度最高的
        best_sampled_point = max(sampled_data, key=lambda x: x['sharpness'])
        best_y = best_sampled_point['position']

        self.message_updated.emit(f"黄金分割搜索结束，最佳采样点位置 {best_y:.3f}mm")

        # 4. 移动到最终最佳点，确认清晰度
        self._update_progress(95)
        self._move_y_absolute_mm(best_y)
        self._wait_settle()

        final_sharpness = self._measure_sharpness_averaged(require_pupil=True)

        if final_sharpness is None or final_sharpness <= 0:
            # 如果检测不到，就回退到采样得到的最大值
            final_sharpness = best_sampled_point['sharpness']
            self.message_updated.emit(f"最终确认失败，使用采样清晰度 {final_sharpness:.2f}")
        else:
            self.message_updated.emit(f"精对焦完成。最终位置: {best_y:.3f}mm, 清晰度: {final_sharpness:.2f}")

        return best_y, final_sharpness

    # 基于爬山法的精对焦方法
    def _fine_focus_climb_hill(self, start_y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        精细对焦（改进的爬山算法）
        """
        self.message_updated.emit(f"开始精对焦，初始位置: {start_y:.2f}mm")

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
                time.sleep(self.config['settle_time'])
                if k == 0:
                    time.sleep(1)
                    k = 1

                sharpness = self._measure_sharpness_averaged(require_pupil=True)
                sharpness_values.append(sharpness if sharpness is not None else 0)

            # 找最佳位置
            max_idx = np.argmax(sharpness_values)
            max_sharpness = sharpness_values[max_idx]

            self.message_updated.emit(
                f"迭代 {iteration + 1}: 步长={step:.3f}mm, "
                f"清晰度=[{sharpness_values[0]:.2f}, {sharpness_values[1]:.2f}, {sharpness_values[2]:.2f}]"
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
                    self.message_updated.emit("达到最小步长，对焦完成")
                    break
            else:
                current_y = positions[max_idx]

            # 早停条件
            if no_improvement_count >= 3:
                self.message_updated.emit("清晰度无明显改善，对焦完成")
                break

        # 移动到最佳位置
        self._move_y_absolute_mm(best_y)
        return best_y, best_sharpness

