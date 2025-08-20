# Autofocusing System for Pupil Detection Cameras

This is an advanced autofocusing system designed for pupil detection cameras used in PhD research for PeriScan. The system integrates computer vision, motor control, and GUI applications to enable automated pupil detection and precise focusing capabilities.

## Overview

The autofocusing system combines sophisticated image processing algorithms with precision motor control to automatically detect and focus on pupils in research environments. It features a state machine-based autofocusing module, real-time image processing with GUI control, and support for professional Basler cameras with DM2C motor drivers.

## Core Components

### Main Applications

- **main.py** - Primary GUI application with PySide6 interface for pupil detection, camera control, and motor alignment
- **auto_focus.py** - State machine-based autofocusing module with coarse/fine focusing algorithms  
- **Autoscan.py** - Simple camera and motor scanning system for image acquisition
- **Motor_Driver.py** - Motor control interface wrapping dm2c library
- **zernike.py** - Zernike polynomial analysis for optical aberrations

### Key Technologies

- **Computer Vision**: OpenCV for image processing with Numba optimization for performance
- **Camera Hardware**: Basler cameras via pypylon library
- **Motor Control**: DM2C motor drivers via Modbus RTU communication
- **GUI Framework**: PySide6 (Qt6) for modern user interfaces
- **Performance**: Numba JIT compilation for optimized image sharpness calculations

## Hardware Requirements

### Camera System
- Basler compatible cameras
- Default exposure: 10,000μs
- Gain range: 0-23
- Support for both manual and automatic exposure/gain modes

### Motor Control System
- DM2C motor drivers with three-axis control:
  - X-axis: Driver_01 (horizontal movement)
  - Y/Z-axis: Driver_02/03 (vertical movement for focusing)
- Communication: Modbus RTU over serial (default COM7, 115200 baud)
- Step conversion: 1mm = 20,000 steps (configurable)
- Default speeds: 800 units/sec with 80 units/sec² acceleration

## Installation

### Prerequisites

Ensure Python 3.x is installed on your system.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `opencv-python==4.12.0.88` - Computer vision processing
- `pypylon==4.2.0` - Basler camera interface
- `PySide6==6.9.1` - Qt GUI framework
- `numba==0.61.2` - JIT compilation for performance optimization
- `numpy` - Numerical computing
- `pyserial==3.5` - Serial communication for motors
- `matplotlib==3.10.5` - Data visualization

## Usage

### Running the Applications

#### Main Pupil Detection System
```bash
python main.py
```
Full GUI application with real-time pupil detection, camera control, and motor alignment capabilities.

#### Simple Scanning System
```bash
python Autoscan.py
```
Streamlined camera and motor scanning for basic image acquisition.

#### Test Zernike Analysis
```bash
python zernike.py
```
Run Zernike polynomial analysis for optical aberration assessment.

#### Run System Tests
```bash
python test.py
```
Execute system validation tests.

### Hardware Setup

1. **Connect Camera**: Ensure Basler camera is properly connected and recognized
2. **Connect Motors**: Connect DM2C motor drivers to serial port (typically COM7)
3. **Launch Application**: Run main.py for full system access
4. **Initialize Hardware**: 
   - Open camera connection and configure parameters
   - Connect to motor controller
5. **Configure Detection**: Enable pupil detection mode
6. **Focus Operation**: Use alignment mode for positioning or autofocus for optimal focus

## Algorithm Details

### Pupil Detection
- Multi-threshold contour detection with parallel processing
- Custom Numba-optimized sharpness calculation functions
- PID controller for precise alignment with configurable parameters

### Autofocus State Machine
1. **Finding Pupil**: Spiral search algorithm to locate pupil within Y-axis range
2. **Coarse Focusing**: Grid search across broader focus range (15 samples default)
3. **Fine Focusing**: Hill-climbing algorithm with adaptive step size
4. **Dual Metrics**: Uses both pupil detection and global sharpness metrics for optimization

### Image Processing Pipeline
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
- Gaussian blur for noise reduction
- Multiple threshold methods (typically 30-60 range)
- Morphological operations for contour cleanup and refinement

## Configuration

### Motor Parameters
- Step conversion ratio configurable in auto_focus.py
- PID parameters adjustable for different system responses
- Focus ranges and search patterns customizable

### Image Processing
- Threshold ranges adjustable for different lighting conditions
- CLAHE parameters tunable for contrast optimization
- Pupil detection sensitivity configurable

### Configuration Files
- **junxi_1.json**: Contains Zernike polynomial coefficients for optical analysis
- **requirements.txt**: Python package dependencies
- Parameters are primarily configured within respective module files

## Data Output

- Images automatically saved to timestamped folders: `scan_YYYYMMDD_HHMMSS/`
- Motor positions internally managed in steps, converted to mm for algorithms
- Real-time processing data available through GUI interfaces

## Research Applications

This system is specifically designed for:
- Pupil tracking and analysis in controlled research environments
- Automated focusing for consistent image quality across sessions
- High-precision positioning for optical measurements
- Integration with PeriScan research protocols

## Development

### Performance Optimization
- Numba JIT compilation for critical image processing functions
- Parallel processing for multi-threshold pupil detection
- Efficient state machine design for real-time autofocusing

### Extensibility
- Modular architecture allows for easy component replacement
- Configurable parameters for adaptation to different hardware setups
- Plugin-ready design for additional image processing algorithms

## Troubleshooting

### Common Issues
- **Camera not detected**: Verify pypylon installation and camera drivers
- **Motor communication failure**: Check COM port settings and cable connections
- **Performance issues**: Ensure Numba is properly installed for JIT optimization
- **GUI display problems**: Verify PySide6 installation and Qt platform plugins

### Hardware Diagnostics
- Use test.py for system validation
- Check motor driver LED indicators for communication status
- Verify camera exposure and gain settings for optimal image quality

## License

This software is developed for PhD research purposes. Please contact the research team for usage permissions and collaboration opportunities.

## Support

For technical support and research collaboration inquiries, please refer to the PeriScan research project documentation or contact the development team.