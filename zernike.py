# 重新执行必要的导入和处理步骤
import json
import numpy as np
import matplotlib.pyplot as plt
import math

# 重新读取 JSON 文件
with open("junxi_1.json", "r") as f:
    data = json.load(f)

# 提取 Zernike 系数（左眼中心注视点第一个测量结果）
zernike_coeffs = data["left"]["points"][1]["result"][0]["zernike"]

# 创建单位圆网格
N = 512
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
mask = R <= 1.0

# 构造 Zernike 多项式函数
def zernike_radial(n, m, r):
    R = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        num = (-1) ** k * math.factorial(n - k)
        denom = (
            math.factorial(k)
            * math.factorial((n + abs(m)) // 2 - k)
            * math.factorial((n - abs(m)) // 2 - k)
        )
        R += num / denom * r ** (n - 2 * k)
    return R

def zernike(n, m, r, theta):
    R = zernike_radial(n, m, r)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(-m * theta)
    else:
        return R

# 生成对应阶次的 Zernike 编号（到 10 阶）
zernike_terms = []
max_order = 10
for n in range(max_order + 1):
    for m in range(-n, n + 1, 2):
        zernike_terms.append((n, m))

# 重建波前图
W = np.zeros_like(R)
for coeff, (n, m) in zip(zernike_coeffs, zernike_terms):
    Z = zernike(n, m, R, Theta)
    Z[~mask] = 0
    W += coeff * Z

# 绘制二维波前图
plt.figure(figsize=(6, 5))
plt.imshow(W, extent=[-1, 1, -1, 1], cmap='jet')
plt.colorbar(label='Wavefront Error (μm)')
plt.title("Wavefront Map from JSON Zernike Coefficients")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.tight_layout()
plt.show()

# 计算统计信息
mean_val = np.mean(W[mask])
std_val = np.std(W[mask])
peak_to_valley = np.max(W[mask]) - np.min(W[mask])

print(f"""std_val:{std_val:.2f}um,"
      mean_val:{mean_val:.2f}um,
      peak_to_valley:{peak_to_valley:.2f}um """)

import math

def zernike_to_refraction(zernike_coeffs, pupil_diameter_mm):
    # 提取第二阶Zernike系数
    c_astig_45 = zernike_coeffs[3]   # Z_2^-2，45°散光项
    c_defocus  = zernike_coeffs[4]   # Z_2^0，离焦项
    c_astig_0  = zernike_coeffs[5]   # Z_2^2，0°/90°散光项

    # 有效瞳孔半径（mm）
    Rp = pupil_diameter_mm / 2.0

    print(c_defocus)
    print(Rp)
    # 计算球镜等效 M（D）
    M = (4 * math.sqrt(3) * c_defocus) / (Rp**2)
    # 由于波前正离焦对应近视，取负号为物理屈光度
    M = -M

    # 计算柱镜峰-谷光度 S_cyl（D）
    S_cyl = (4 * math.sqrt(6) * math.hypot(c_astig_0, c_astig_45)) / (Rp**2)
    # 验光惯例取负柱镜
    cyl = -S_cyl

    # 计算柱镜轴向（°）
    theta_max = 0.5 * math.atan2(c_astig_45, c_astig_0)  # 最大屈光力方向
    axis = math.degrees(theta_max + math.pi/2) % 180    # 柱镜轴 = 最大经线 + 90°

    # 将球镜等效拆分为 Sphere 和 Cylinder
    sphere = M + S_cyl/2
    cylinder = cyl
    axis_deg = axis

    return sphere, cylinder, axis_deg

sphere, cylinder, axis = zernike_to_refraction(zernike_coeffs, pupil_diameter_mm=(3.925 + 3.214)/2)
print(f"Sphere = {sphere:.3f} D, Cylinder = {cylinder:.3f} D, Axis = {axis:.1f}°")

