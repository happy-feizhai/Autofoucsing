import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math
import json
import os


def load_zernike_from_json(json_file_path, eye='left', offaxis=0, meridian=0, use_average=True):
    """
    Load Zernike coefficients from JSON file

    Parameters:
    json_file_path: Path to the JSON file
    eye: 'left' or 'right'
    offaxis: Off-axis angle (0, 15, 30, 45, etc.)
    meridian: Meridian angle (0, 90, 180, 270)
    use_average: If True, use average values; if False, use median values

    Returns:
    zernike_coefficients: numpy array of Zernike coefficients
    measurement_info: dictionary with measurement details
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get eye data
        if eye not in data:
            raise ValueError(f"Eye '{eye}' not found in data. Available: {list(data.keys())}")

        eye_data = data[eye]

        # Find the matching point
        target_point = None
        for point in eye_data['points']:
            if point['offaxis'] == offaxis and point['meridian'] == meridian:
                target_point = point
                break

        if target_point is None:
            raise ValueError(f"No measurement found for offaxis={offaxis}, meridian={meridian}")

        # Check if there are any results
        if not target_point['result']:
            raise ValueError(f"No measurement results for offaxis={offaxis}, meridian={meridian}")

        # Get Zernike coefficients
        if use_average and 'average' in target_point and target_point['average']:
            zernike_coeffs = target_point['average']['zernike']
            data_type = 'average'
        # elif not use_average and 'median' in target_point and target_point['median']:
        #     zernike_coeffs = target_point['result'][0]['zernike']
        #     data_type = 'median'
        else:
            # Use the first measurement if average/median not available
            zernike_coeffs = target_point['result'][3]['zernike']
            data_type = 'first_measurement'
            print(f"Warning: Using first measurement result instead of {data_type}")

        # Create measurement info
        measurement_info = {
            'patient_name': data.get('name', 'Unknown'),
            'date': data.get('date', 'Unknown'),
            'eye': eye,
            'offaxis': offaxis,
            'meridian': meridian,
            'data_type': data_type,
            'num_measurements': len(target_point['result']),
            'file_path': json_file_path
        }

        # Add clinical parameters if available
        if use_average and 'average' in target_point and target_point['average']:
            clinical_data = target_point['average']
        # elif not use_average and 'median' in target_point and target_point['median']:
        #     clinical_data = target_point['median']
        else:
            clinical_data = target_point['result'][3]

        measurement_info.update({
            'wf_rms': clinical_data.get('wf_rms', 0),
            'wf_pv': clinical_data.get('wf_pv', 0),
            'sphere': clinical_data.get('sphere', 0),
            'cyl': clinical_data.get('cyl', 0),
            'axis': clinical_data.get('axis', 0)
        })

        return np.array(zernike_coeffs), measurement_info

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_file_path}")
    except Exception as e:
        raise Exception(f"Error loading Zernike coefficients: {str(e)}")


def list_available_measurements(json_file_path):
    """
    List all available measurements in the JSON file

    Parameters:
    json_file_path: Path to the JSON file

    Returns:
    Dictionary with available measurements for each eye
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        available_measurements = {}

        for eye in ['left', 'right']:
            if eye in data:
                measurements = []
                for point in data[eye]['points']:
                    if point['result']:  # Only include points with actual measurements
                        measurements.append({
                            'offaxis': point['offaxis'],
                            'meridian': point['meridian'],
                            'num_results': len(point['result']),
                            'has_average': bool(point.get('average')),
                            'has_median': bool(point.get('median'))
                        })
                available_measurements[eye] = measurements

        return available_measurements

    except Exception as e:
        print(f"Error listing measurements: {str(e)}")
        return {}


def zernike_polynomial(n, m, rho, theta):
    """
    ANSI Z80.28-2010标准定义Zernike多项式 (包含正确归一化)
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)

    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        R += ((-1)**k * math.factorial(n - k) /
              (math.factorial(k) *
               math.factorial((n + abs(m))//2 - k) *
               math.factorial((n - abs(m))//2 - k))) * rho**(n - 2*k)

    # 引入标准ANSI归一化因子
    if m == 0:
        if n == 0:
            Z = 0
            return Z
        norm_factor = math.sqrt(n + 1)
        Z = norm_factor * R
    elif m > 0:
        norm_factor = math.sqrt(2 * (n + 1))
        Z = norm_factor * R * np.cos(m * theta)
    else:
        norm_factor = math.sqrt(2 * (n + 1))
        Z = norm_factor * R * np.sin(abs(m) * theta)

    return Z


def get_zernike_index(j):
    """
    Get (n,m) from single index j
    Standard Zernike polynomial index conversion
    """
    n = int((-1 + np.sqrt(1 + 8 * j)) / 2)
    m = 2 * j - n * (n + 2)
    return n, m


def generate_wavefront_error(coefficients, grid_size=256):
    """
    Generate wavefront error map based on Zernike coefficients

    Parameters:
    coefficients: Zernike coefficient array (starting from Z0)
    grid_size: Grid size

    Returns:
    x, y: Coordinate grids
    wavefront: Wavefront error array
    """
    # 创建圆形网格
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # 转换为极坐标
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    # 创建圆形mask
    mask = rho <= 1.0

    # 初始化波前误差
    wavefront = np.zeros_like(rho)

    # 计算每个泽尼克项的贡献
    for j, coeff in enumerate(coefficients):
        if coeff != 0:
            n, m = get_zernike_index(j)
            zernike = zernike_polynomial(n, m, rho, theta)
            wavefront += coeff * zernike

    # 应用圆形mask
    wavefront[~mask] = np.nan

    return X, Y, wavefront


def get_zernike_names_english():
    """
    Get standard English names of Zernike polynomials
    """
    names = []
    # Add generic names for higher order terms
    for i in range(0, 100):
        n, m = get_zernike_index(i)
        names.append(f"Z{i}: n={n},m={m}")

    return names


def calculate_thorlabs_rms(zernike_coefficients):
    """
    根据Thorlabs WFS用户手册计算RMS值 (不包含活塞项)
    参数:
        zernike_coefficients: Zernike系数数组
    返回:
        rms_total: RMS误差（μm）
    """
    # 排除活塞项（第一个系数 Z0）
    squared_coeffs = np.array(zernike_coefficients[1:]) ** 2
    total_squared = np.sum(squared_coeffs)
    rms_total = np.sqrt(total_squared)

    return rms_total

def analyze_wavefront_statistics(wavefront):
    """
    分析波前误差统计信息
    """
    # 移除 NaN 值
    valid_data = wavefront[~np.isnan(wavefront)]

    print("波前误差统计分析:")
    # 修正: 使用标准 RMS（不去除均值）计算波前误差
    rms_val = np.sqrt(np.mean(valid_data ** 2))
    print(f"RMS误差: {rms_val:.3f} μm")
    print(f"P-V值 (Peak-to-Valley): {np.max(valid_data) - np.min(valid_data):.3f} μm")
    print(f"最大值: {np.max(valid_data):.3f} μm")
    print(f"最小值: {np.min(valid_data):.3f} μm")
    print(f"平均值: {np.mean(valid_data):.3f} μm")


def comprehensive_analysis(coefficients, grid_size=256, save_path=None):
    """
    Comprehensive analysis: Display wavefront error and coefficient analysis in one window

    Parameters:
    coefficients: Zernike coefficient array
    grid_size: Grid size
    save_path: Save path (optional)
    """
    # Set font for better display
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # Generate wavefront error
    X, Y, wavefront = generate_wavefront_error(coefficients, grid_size)

    # Create comprehensive figure (2 rows, 3 columns layout)
    fig = plt.figure(figsize=(18, 10))

    # 1. Wavefront error contour plot
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.contourf(X, Y, wavefront, levels=50, cmap='RdYlBu_r')
    ax1.set_aspect('equal')
    ax1.set_title('Wavefront Error Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x (normalized)', fontsize=10)
    ax1.set_ylabel('y (normalized)', fontsize=10)
    circle1 = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle1)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Wavefront Error (μm)', rotation=270, labelpad=15, fontsize=9)

    # 2. All coefficients bar chart
    ax2 = plt.subplot(2, 3, 2)
    indices = np.arange(len(coefficients))
    colors = ['red' if c < 0 else 'blue' for c in coefficients]
    bars2 = ax2.bar(indices, coefficients, color=colors, alpha=0.7)
    ax2.set_xlabel('Zernike Term Index', fontsize=10)
    ax2.set_ylabel('Coefficient Value (μm)', fontsize=10)
    ax2.set_title('Zernike Coefficient Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Only show labels for larger coefficients to avoid crowding
    max_coeff = np.max(np.abs(coefficients))
    for i, (bar, coeff) in enumerate(zip(bars2, coefficients)):
        if abs(coeff) > max_coeff * 0.3:  # Only label coefficients > 30% of max value
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + np.sign(height) * max_coeff * 0.02,
                     f'{coeff:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=7, rotation=90)

    # 3. RMS contribution by order
    ax3 = plt.subplot(2, 3, 3)
    max_order = 10
    order_rms = []
    order_labels = []

    for order in range(max_order + 1):
        # Find all terms of this order
        order_indices = []
        for j in range(len(coefficients)):
            n, m = get_zernike_index(j)
            if n == order:
                order_indices.append(j)

        # 排除活塞项 Z0（通常是 j=0）
        order_indices = [i for i in order_indices if i != 0]


        if order_indices:
            order_coeffs = [coefficients[i] for i in order_indices if i < len(coefficients)]
            rms_value = np.sqrt(np.mean(np.array(order_coeffs) ** 2))
            order_rms.append(rms_value)
            order_labels.append(f'Order {order}')

    bars3 = ax3.bar(range(len(order_rms)), order_rms, color='green', alpha=0.7)
    ax3.set_xlabel('Zernike Order', fontsize=10)
    ax3.set_ylabel('RMS Value (μm)', fontsize=10)
    ax3.set_title('RMS Contribution by Order', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(order_rms)))
    ax3.set_xticklabels([f'{i}' for i in range(len(order_rms))], fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, rms_val in zip(bars3, order_rms):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + max(order_rms) * 0.02,
                 f'{rms_val:.3f}', ha='center', va='bottom', fontsize=8)

    # 4. Radial profile analysis
    ax4 = plt.subplot(2, 3, 4)
    # Extract radial profiles (horizontal and vertical directions)
    center_y = wavefront.shape[0] // 2
    center_x = wavefront.shape[1] // 2

    # Horizontal profile (from center to right)
    horizontal_profile = wavefront[center_y, center_x:]
    horizontal_profile = horizontal_profile[~np.isnan(horizontal_profile)]

    # Vertical profile (from center upward)
    vertical_profile = wavefront[center_y:, center_x]
    vertical_profile = vertical_profile[~np.isnan(vertical_profile)]

    radius_h = np.linspace(0, 1, len(horizontal_profile))
    radius_v = np.linspace(0, 1, len(vertical_profile))

    ax4.plot(radius_h, horizontal_profile, 'b-', label='Horizontal Profile', linewidth=2, marker='o', markersize=2)
    ax4.plot(radius_v, vertical_profile, 'r--', label='Vertical Profile', linewidth=2, marker='s', markersize=2)
    ax4.set_xlabel('Normalized Radial Distance', fontsize=10)
    ax4.set_ylabel('Wavefront Error (μm)', fontsize=10)
    ax4.set_title('Radial Profile Analysis', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # Add symmetry analysis
    if len(horizontal_profile) == len(vertical_profile):
        correlation = np.corrcoef(horizontal_profile, vertical_profile)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=ax4.transAxes, fontsize=9, verticalalignment='top')

    # 5. Wavefront statistics and key parameters
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    # 修正: 计算标准RMS（包含波前平均值偏移）
    rms_error = np.sqrt(np.nanmean(wavefront ** 2))  # 标准 RMS
    pv_value = np.nanmax(wavefront) - np.nanmin(wavefront)
    total_rms_coeff = np.sqrt(np.sum(coefficients[1:]** 2))
    non_zero_count = np.sum(np.abs(coefficients) > 1e-6)


    stats_text = (
        f"• RMS Error: {rms_error:.3f} μm\n"
        f"• P-V Value: {pv_value:.3f} μm  \n"
        f"• Maximum: {np.nanmax(wavefront):.3f} μm\n"
        f"• Minimum: {np.nanmin(wavefront):.3f} μm\n\n"
        f"• Total RMS: {total_rms_coeff:.3f} μm\n"
        f"• Number of effective terms: {non_zero_count}\n"
        f"• Largest contribution: "
    )
    # Find the term with largest absolute coefficient
    if len(coefficients) > 1:
        # 排除第一个活塞项 (索引0) 再寻找最大贡献项
        max_contrib_idx = int(np.argmax(np.abs(coefficients[1:]))) + 1  # 加1修正索引
        max_contrib_name = get_zernike_names_english()[max_contrib_idx] if max_contrib_idx < len(
            get_zernike_names_english()) else f"Z{max_contrib_idx}"
        stats_text += f"{max_contrib_name}\n  Coefficient: {coefficients[max_contrib_idx]:.3f} μm"
    else:
        stats_text += "N/A"

    ax5.text(0.5, 0.5, stats_text, fontsize=10, color="blue", va='bottom')


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_zernike_coefficients(coefficients):
    """
    分析泽尼克系数统计信息
    """
    print("\n泽尼克系数分析:")
    print(f"总RMS: {np.sqrt(np.sum(coefficients[1:] ** 2)):.3f} μm")
    print(f"非零项数量: {np.sum(np.abs(coefficients) > 1e-6)}")

    # 找出贡献最大的前5项
    sorted_indices = np.argsort(np.abs(coefficients[1:]))[::-1]
    print("\n贡献最大的前5项:")
    names = get_zernike_names_english()
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i] + 1
        if abs(coefficients[idx]) > 1e-6:
            name = names[idx] if idx < len(names) else f"Z{idx}"
            print(f"  {name}: {coefficients[idx]:.3f} μm")

    # 各阶的RMS贡献
    print("\n各阶RMS贡献:")
    for order in range(6):  # 显示前6阶
        order_indices = []
        for j in range(len(coefficients[1:])):
            n, m = get_zernike_index(j)
            if n == order:
                order_indices.append(j)

        if order_indices and order != 0:
            order_coeffs = [coefficients[i] for i in order_indices if i < len(coefficients) ]
            rms_value = np.sqrt(np.mean(np.array(order_coeffs) ** 2))
            print(f"  {order}阶: {rms_value:.3f} μm")



# Example usage
if __name__ == "__main__":
    # Configuration - Set USE_JSON_FILE to True to load from JSON
    USE_JSON_FILE = True
    JSON_FILE_PATH = "junxi_1.json"  # Change this to your JSON file path

    if USE_JSON_FILE and os.path.exists(JSON_FILE_PATH):
        try:
            print("Loading Zernike coefficients from JSON file...")
            print(f"File: {JSON_FILE_PATH}")

            # List available measurements
            available = list_available_measurements(JSON_FILE_PATH)
            print("\nAvailable measurements:")
            for eye, measurements in available.items():
                print(f"\n{eye.upper()} EYE:")
                for i, meas in enumerate(measurements):
                    print(f"  {i + 1}. Offaxis: {meas['offaxis']}°, Meridian: {meas['meridian']}°, "
                          f"Results: {meas['num_results']}, "
                          f"Average: {'✓' if meas['has_average'] else '✗'}, "
                          f"Median: {'✓' if meas['has_median'] else '✗'}")

            # Load specific measurement (you can modify these parameters)
            eye = 'left'  # 'left' or 'right'
            offaxis = 0  # 0, 15, 30, 45, etc.
            meridian = 0  # 0, 90, 180, 270
            use_average = False  # True for average, False for median

            zernike_coefficients, measurement_info = load_zernike_from_json(
                JSON_FILE_PATH, eye=eye, offaxis=offaxis, meridian=meridian, use_average=use_average
            )

            print(f"\nLoaded measurement for:")
            print(f"  Patient: {measurement_info['patient_name']}")
            print(f"  Date: {measurement_info['date']}")
            print(f"  Eye: {measurement_info['eye'].upper()}")
            print(f"  Position: {measurement_info['offaxis']}° off-axis, {measurement_info['meridian']}° meridian")
            print(f"  Data type: {measurement_info['data_type']}")
            print(f"  Number of measurements: {measurement_info['num_measurements']}")
            print(f"  Clinical data:")
            print(f"    - RMS: {measurement_info['wf_rms']:.3f} μm")
            print(f"    - P-V: {measurement_info['wf_pv']:.3f} μm")
            print(f"    - Sphere: {measurement_info['sphere']:.3f} D")
            print(f"    - Cylinder: {measurement_info['cyl']:.3f} D")
            print(f"    - Axis: {measurement_info['axis']:.1f}°")
            print(f"  Zernike coefficients: {len(zernike_coefficients)} terms")

        except Exception as e:
            print(f"Error loading JSON file: {e}")
            print("Using example data instead...")
            USE_JSON_FILE = False

    if not USE_JSON_FILE:
        print("Using example Zernike coefficients...")

        # Example: 10th order Zernike coefficients
        zernike_coefficients = np.array([
            0.000,  # Z0: Piston
            0.150,  # Z1: x-tilt
            0.120,  # Z2: y-tilt
            0.250,  # Z3: Defocus
            0.080,  # Z4: 45° astigmatism
            0.060,  # Z5: 0° astigmatism
            0.040,  # Z6: y-trefoil
            0.030,  # Z7: x-trefoil
            0.020,  # Z8: y-secondary astigmatism
            0.015,  # Z9: Primary spherical aberration
            # Continue adding more coefficients up to 10th order...
            0.010, 0.008, 0.006, 0.005, 0.004,  # Z10-Z14
            0.003, 0.002, 0.001, 0.001, 0.001,  # Z15-Z19
            0.001, 0.001, 0.001, 0.001, 0.001,  # Z20-Z24
            0.001, 0.001, 0.001, 0.001, 0.001,  # Z25-Z29
            0.001, 0.001, 0.001, 0.001, 0.001,  # Z30-Z34
            0.001, 0.001, 0.001, 0.001, 0.001,  # Z35-Z39
            0.001, 0.001, 0.001, 0.001, 0.001,  # Z40-Z44
            0.001, 0.001, 0.001, 0.001, 0.001,  # Z45-Z49
            0.001, 0.001, 0.001, 0.001, 0.001  # Z50-Z54
        ])

        measurement_info = {
            'patient_name': 'Example',
            'eye': 'example',
            'data_type': 'synthetic'
        }

    print("\nGenerating human eye wavefront error comprehensive analysis report...")

    # Generate comprehensive analysis report
    comprehensive_analysis(zernike_coefficients, grid_size=512)

    # Output text statistical analysis
    print("\n" + "=" * 50)
    X, Y, wavefront = generate_wavefront_error(zernike_coefficients, grid_size=512)
    analyze_wavefront_statistics(wavefront)
    analyze_zernike_coefficients(zernike_coefficients)
    print("=" * 50)


    rms = calculate_thorlabs_rms(zernike_coefficients)

    print(f"Thorlabs标准RMS误差为: {rms:.6f} μm")

    # If you need to save results
    # comprehensive_analysis(zernike_coefficients, save_path="comprehensive_analysis.png")

