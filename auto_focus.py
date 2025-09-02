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

        # 对焦参数（可调整）
        self.config = {
            # 瞳孔搜索参数
            'pupil_search_range': 80.0,  # mm
            'pupil_search_coarse_step': 3.0,  # mm
            'pupil_search_fine_step': 1,  # mm

            # 粗对焦参数
            'coarse_range': 20.0,  # mm
            'coarse_samples': 12,  # 采样点数

            # 精对焦参数
            'fine_range': 1.0,  # mm
            'fine_initial_step': 1.2,  # mm
            'fine_min_step': 0.1,  # mm
            'fine_max_iterations': 30,

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
                self._handle_focus_failed("粗对焦失败")
                return

            self.message_updated.emit(f"粗对焦完成，最佳位置: {best_coarse_z:.2f}mm")

            # 阶段3：精对焦
            self._change_state(FocusState.FINE_FOCUSING)
            self._update_progress(60)

            final_y, final_sharpness = self._fine_focus(best_coarse_z)

            if self.cancel_requested:
                self._handle_cancel()
                return

            if final_y is None:
                self._handle_focus_failed("精对焦失败")
                return

            # 对焦成功
            self._change_state(FocusState.FOCUSED)
            self._update_progress(100)

            elapsed_time = time.time() - start_time
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
            if i == 1:
                time.sleep(0.5)

            if self._check_pupil_detection():
                pupil_positions.append(y_pos)

                if i == 0:
                    # 中心点就检测到瞳孔，直接返回一个小范围
                    return y_pos - 10, y_pos + 10

                # 找到瞳孔后，在附近细化搜索边界
                if len(pupil_positions) == 1:
                    return y_pos - 10, y_pos + 10
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
        粗对焦搜索
        """
        # TODO: 改进粗对焦清晰度搜索方法，如果清晰度下降就立即停止，并以下降前的最高位置作为精对焦的初始位置
        self.message_updated.emit(f"粗对焦搜索: {y_min:.2f}mm 到 {y_max:.2f}mm")

        n_samples = self.config['coarse_samples']
        positions = np.linspace(y_min, y_max, n_samples)
        results = []

        for i, y_pos in enumerate(positions):
            if self.cancel_requested:
                return None
            self._update_progress(30 + int(30 * i / len(positions)))

            self._move_y_absolute_mm(y_pos)
            time.sleep(self.config['settle_time'])
            if i == 0:
                time.sleep(0.5)

            # 计算清晰度（多帧平均）
            sharpness = self._measure_sharpness_averaged()

            results.append({
                'position': y_pos,
                'sharpness': sharpness if sharpness is not None else 0,
                'has_pupil': self._check_pupil_detection()
            })

            self.message_updated.emit(f"位置 {y_pos:.2f}mm, 清晰度: {sharpness:.2f}")

        # 选择最佳位置（优先选择检测到瞳孔的位置）
        pupil_results = [r for r in results if r['has_pupil']]

        if pupil_results:
            best = max(pupil_results, key=lambda x: x['sharpness'])
        elif results:
            best = max(results, key=lambda x: x['sharpness'])
        else:
            return None

        # 可选：使用二次拟合预测更精确的峰值位置
        # if len(results) >= 3:
        #     predicted = self._predict_peak_position(results)
        #     if predicted and y_min <= predicted <= y_max:
        #         return predicted
        #
        return best['position']

    def _fine_focus(self, start_y: float) -> Tuple[Optional[float], Optional[float]]:
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
        check_frames = 5

        for _ in range(check_frames):
            img = self.get_image()
            pupil_circle, _ = self.detect_pupil(img)
            if pupil_circle is not None:
                detected_count += 1

        return detected_count >= check_frames * self.config['pupil_detect_threshold']

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