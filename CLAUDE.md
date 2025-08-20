# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an autofocusing system for pupil detection cameras used in PhD research for PeriScan. The system combines computer vision, motor control, and GUI applications for automated pupil detection and focusing.

## Core Architecture

### Main Components

- **main.py** - Primary GUI application with PySide6 interface for pupil detection, camera control, and motor alignment
- **auto_focus.py** - State machine-based autofocusing module with coarse/fine focusing algorithms
- **Autoscan.py** - Simple camera and motor scanning system for image acquisition
- **Motor_Driver.py** - Motor control interface wrapping dm2c library
- **zernike.py** - Zernike polynomial analysis for optical aberrations (used with junxi_1.json data)

### Key Technologies

- **Computer Vision**: OpenCV for image processing, Numba for performance optimization
- **Camera Hardware**: Basler cameras via pypylon library  
- **Motor Control**: DM2C motor drivers via Modbus RTU (typically COM7)
- **GUI Framework**: PySide6 (Qt6) for user interfaces
- **Optimization**: Numba JIT compilation for image sharpness calculations

### Motor Control System

The system uses DM2C motor drivers with three axes:
- X-axis: Driver_01 (horizontal movement)
- Y/Z-axis: Driver_02/03 (vertical movement for focusing)
- Communication via Modbus RTU over serial (default COM7, 115200 baud)

## Common Development Commands

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Applications

```bash
# Main pupil detection system
python temp.py

# Simple scanning system
python Autoscan.py

# Test Zernike polynomial analysis
python zernike.py

# Run tests
python test.py
```

## Key Dependencies

From requirements.txt:
- opencv-python==4.12.0.88 (computer vision)
- pypylon==4.2.0 (Basler camera interface)
- PySide6==6.9.1 (Qt GUI framework)
- numba==0.61.2 (JIT compilation for performance)
- numpy (numerical computing)
- pyserial==3.5 (motor communication)
- matplotlib==3.10.5 (visualization)

## Hardware Configuration

### Camera Parameters
- Default exposure: 10000μs
- Gain: 0-23 range
- Supports both manual and auto exposure/gain modes

### Motor Parameters  
- Step conversion: 1mm = 20000 steps (configurable in auto_focus.py)
- Default speeds: 800 units/sec
- Acceleration/deceleration: 80 units/sec²

## Algorithm Details

### Pupil Detection
- Uses multi-threshold contour detection with parallel processing
- Sharpness calculated via custom Numba-optimized function
- PID controller for alignment with configurable parameters

### Autofocus State Machine
1. **Finding Pupil**: Spiral search to locate pupil in Y-axis range
2. **Coarse Focusing**: Grid search across broader range (15 samples default)  
3. **Fine Focusing**: Hill-climbing algorithm with adaptive step size
4. Uses both pupil detection and global sharpness metrics

### Image Processing Pipeline
- CLAHE contrast enhancement
- Gaussian blur noise reduction  
- Multiple threshold methods (typically 30-60 range)
- Morphological operations for contour cleanup

## Configuration Files

- **junxi_1.json**: Contains Zernike polynomial coefficients for optical analysis
- **requirements.txt**: Python package dependencies
- No dedicated config files - parameters are hard-coded in respective modules

## Typical Workflow

1. Connect hardware (camera + motor on COM7)
2. Run main.py for full system
3. Open camera and set parameters
4. Connect motor controller
5. Enable pupil detection mode
6. Use alignment mode for positioning or autofocus for optimal focus

## Development Notes

- Images saved to timestamped folders (scan_YYYYMMDD_HHMMSS/)
- Motor positions internally use steps, converted to mm in algorithms
- PID parameters and focus ranges are configurable in auto_focus.py config dict
- Pupil detection optimized for performance with Numba JIT compilation