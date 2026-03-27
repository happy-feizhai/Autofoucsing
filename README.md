# Autofocusing System for PeriScan

This repository contains a camera-based autofocus and pupil-alignment system used in PeriScan research. It combines Basler image acquisition, pupil detection, motor control (DM2C + LMS servo), and a PySide6 GUI.

## Current Project Layout

- `temp.py`: **primary entry point** for the current workflow (GUI, pupil detection, alignment, autofocus integration).
- `auto_focus.py`: autofocus state machine (`AutoFocusStateMachine`) with pupil search and fine-focus logic.
- `Motor_Driver.py`: motion abstraction layer. X/Z axes use DM2C stepping drivers; Y axis is adapted to LMS servo (`DM2C.Driver_02 -> LMS_LinearMovement`).
- `dm2c.py`, `lms_drive.py`: low-level Modbus RTU communication and driver operations.
- `Autoscan.py`: step-scan capture utility, saves images under `scan_YYYYMMDD_HHMMSS/`.
- `compare.py`: offline pupil-detection visualization, outputs to `pupil_results/<timestamp>/`.
- `lms_test.py`: LMS debug GUI (connect, relative move, fault reset, status query).
- `zernike_test.py`, `zernike.py`: Zernike/wavefront analysis using `junxi_1.json`.
- `main.py`, `test.py`: older GUI/test scripts; still runnable, but `temp.py` is recommended.

## Requirements

- Python `3.12.9` (see `.python-version` and `pyproject.toml`).
- Basler camera with `pypylon`.
- Motor controllers connected over serial (default `COM7` for main app, `COM8` in `lms_test.py`).

## Installation

```bash
# Option A (recommended, aligns with pyproject.toml)
uv sync

# Option B
python -m pip install -r requirements.txt
python -m pip install scipy
```

Note: `auto_focus.py` uses `scipy.optimize.curve_fit`, but `requirements.txt` does not currently include `scipy`.

## Quick Start

```bash
python temp.py
```

Typical sequence:

1. Open the Basler camera.
2. Connect motors (`COM7`, `115200` by default).
3. Enable pupil detection (optional: pupil alignment).
4. Start autofocus from the GUI.

## Autofocus Logic (Current Implementation)

`auto_focus.py::_run_focus_sequence()` performs:

1. Enable Y-axis servo in PP mode (fault reset + operation enable).
2. Check if pupil is detectable at current position.
3. If not detectable, search along Y within configured range.
4. Apply a relative offset adjustment.
5. Run fine focus (default path uses `_fine_focus_continuous_pp()`), estimate best peak position, and move back to that point.

Unit conversion used by autofocus:

- `1 mm = 20000 steps` (`steps_per_mm` in `auto_focus.py`).

## Other Useful Scripts

```bash
python Autoscan.py      # camera + motor scanning capture
python lms_test.py      # LMS servo diagnostics UI (default COM8)
python compare.py       # offline pupil-detection visualization
python zernike_test.py  # comprehensive Zernike analysis
python zernike.py       # script-style wavefront/refraction calculation
```

## Key Configuration Locations

- Motor axis parameters and serial config: `temp.py` (`MotorController.AXIS_CONFIG`, `SERIAL_PORT`, `BAUDRATE`).
- Autofocus algorithm parameters: `auto_focus.py` (`self.config`).
- GUI focus-search range: `temp.py` (`search_range_spin`).

## Troubleshooting

- Qt plugin path is set in scripts via `QT_QPA_PLATFORM_PLUGIN_PATH`.
- If `scipy` import fails, install it manually.
- For serial communication issues, verify COM port and baudrate first.
- For LMS faults, use `lms_test.py` and run Fault Reset.

## Notes

- This is a hardware-coupled project; many functions require a connected camera and motor controller.
- `zernike.py` currently executes analysis code at top level (script style), while `zernike_test.py` provides a more complete analysis workflow.
