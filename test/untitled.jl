using HopTB
using Plots
using Colors
lat = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
BN = TBModel(2, lat)
# 子晶格 A 的位置 (在晶格点 [0, 0, 0])
setposition!(BN, [0, 0, 0], 1, 1, 1, (BN.lat * [0, 0, 0])[1])
setposition!(BN, [0, 0, 0], 1, 1, 2, (BN.lat * [0, 0, 0])[2])
setposition!(BN, [0, 0, 0], 1, 1, 3, (BN.lat * [0, 0, 0])[3])

# 子晶格 B 的位置 (在晶格点 [a/2, a/2, 0])
setposition!(BN, [0, 0, 0], 2, 2, 1, (BN.lat * [1/2, 1/2, 0])[1])
setposition!(BN, [0, 0, 0], 2, 2, 2, (BN.lat * [1/2, 1/2, 0])[2])
setposition!(BN, [0, 0, 0], 2, 2, 3, (BN.lat * [1/2, 1/2, 0])[3])


# Parameters
t = 0.15
λ =0.4
J1 = 1
J2 = 0
# 1. -t(cos(kx) + cos(ky)) * σ0
# 作用在哈密顿量的对角线元素上
sethopping!(BN, [1, 0, 0], 1, 1, -t)  # kx方向上的跳跃 (σ0)
sethopping!(BN, [-1, 0, 0], 1, 1, -t) # 反向的kx跳跃 (σ0)
sethopping!(BN, [1, 0, 0], 2, 2, -t)  # kx方向上的跳跃 (σ0)
sethopping!(BN, [-1, 0, 0], 2, 2, -t) # 反向的kx跳跃 (σ0)
sethopping!(BN, [0, 1, 0], 2, 2, -t)  # ky方向上的跳跃 (σ0)
sethopping!(BN, [0, -1, 0], 2, 2, -t) # 反向的ky跳跃 (σ0)
sethopping!(BN, [0, 1, 0], 1, 1, -t)  # ky方向上的跳跃 (σ0)
sethopping!(BN, [0, -1, 0], 1, 1, -t) # 反向的ky跳跃 (σ0)


# 2. (λ/2) * [sin(kx + ky) * σx + sin(ky - kx) * σy]
# 作用在哈密顿量的非对角线元素上 (σx, σy)
sethopping!(BN, [1, 1, 0], 1, 2, -(λ/2)*im)# kx+ky方向上的跳跃 (σx)
sethopping!(BN, [-1, -1, 0], 1, 2, -(-λ/2)*im)# 反向跳跃 (σx)
sethopping!(BN, [-1, -1, 0], 2, 1, -(-λ/2)*im) # 反向跳跃 (σx)
sethopping!(BN, [1, 1, 0], 2, 1, -(λ/2)*im)  # kx+ky方向上的跳跃 (σx)

sethopping!(BN, [1, -1, 0], 1, 2, λ/2) # kx-ky方向上的跳跃 (σy)
sethopping!(BN, [-1, 1, 0], 1, 2, -λ/2) # 反向跳跃 (σy)
sethopping!(BN, [1, -1, 0], 2, 1, -λ/2) # kx-ky方向上的跳跃 (σy)
sethopping!(BN, [-1, 1, 0], 2, 1, λ/2) # 反向跳跃 (σy)

# 3. J1 * (cos(kx) - cos(ky)) + J2 * sin(kx) * sin(ky) * σz
# 作用在哈密顿量的对角线元素上 (σz)
addhopping!(BN, [1, 0, 0], 1, 1, J1)   # kx方向上的跳跃 (σz)
addhopping!(BN, [-1, 0, 0], 1, 1, J1)   # kx方向上的跳跃 (σz)
addhopping!(BN, [0, 1, 0], 1, 1, -J1)  # ky方向上的跳跃 (σz)
addhopping!(BN, [0, -1, 0], 1, 1, -J1)  # ky方向上的跳跃 (σz)
addhopping!(BN, [1, 0, 0], 2, 2, -J1)  # kx方向上的跳跃 (σz)
addhopping!(BN, [-1, 0, 0], 2, 2, -J1)   # kx方向上的跳跃 (σz)
addhopping!(BN, [0, 1, 0], 2, 2, J1)  # ky方向上的跳跃 (σz)
addhopping!(BN, [0, -1, 0], 2, 2, J1)  # ky方向上的跳跃 (σz)
addhopping!(BN, [1, 1, 0], 1, 1, -J2)    # kx+ky方向的跳跃 (σz)
addhopping!(BN, [-1, -1, 0], 1, 1, -J2)  # 反向跳跃 (σz)
addhopping!(BN, [1, -1, 0], 1, 1, J2)    # kx+ky方向的跳跃 (σz)
addhopping!(BN, [-1, 1, 0], 1, 1, J2)  # 反向跳跃 (σz)
addhopping!(BN, [1, 1, 0], 2, 2, J2)    # kx+ky方向的跳跃 (σz)
addhopping!(BN, [-1, -1, 0], 2, 2, J2)  # 反向跳跃 (σz)
addhopping!(BN, [1, -1, 0], 2, 2, -J2)    # kx+ky方向的跳跃 (σz)
addhopping!(BN, [-1, 1, 0], 2, 2, -J2)  # 反向跳跃 (σz)





# 定义 kx 和 ky 的取值范围
kx_range = range(-0.4, 0.4, length=100)
ky_range = range(-0.4, 0.4, length=100)

# 生成 kx 和 ky 的一维数组
kx = vec(kx_range)
ky = vec(ky_range)

# 创建 kx 和 ky 网格
kx_grid = [kx[i] for i in 1:length(kx), j in 1:length(ky)]
ky_grid = [ky[j] for i in 1:length(kx), j in 1:length(ky)]

# 将 kx 和 ky 转换为一维数组，并生成 kpts 矩阵
kx_flat = kx_grid[:]
ky_flat = ky_grid[:]
kz_flat = zeros(length(kx_flat))  # 假设 kz = 0

kpts = hcat(kx_flat, ky_flat, kz_flat)'
berry_curvatures = Float64[]
# 计算 Berry 曲率
for i in 1:size(kpts, 2)
    k = kpts[:, i]
    curvature = get_berry_curvature(BN, 2, 1, k)[1]  # 假设我们只需要第一条能带的 Berry 曲率
    push!(berry_curvatures, curvature)
end
# 定义颜色范围和边界
# 生成图像
p = scatter(
    kx_flat, ky_flat,zcolor=berry_curvatures,
    color=:viridis,# 选择一个色图
    xlabel="kx",
    ylabel="ky",
    title="Berry Curvature in kx-ky Plane",
    clims=(-0.5, 0.5)  # 设置颜色条的范围
)

# 显示图像
display(p)