using HopTB
using PyPlot
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
addhopping!(BN, [1, 1, 0], 1, 2, λ/2/im )# kx+ky方向上的跳跃 (σx)
addhopping!(BN, [-1, -1, 0], 1, 2, -λ/2/im)# 反向跳跃 (σx)
addhopping!(BN, [-1, -1, 0], 2, 1, -λ/2/im) # 反向跳跃 (σx)
addhopping!(BN, [1, 1, 0], 2, 1, λ/2/im)  # kx+ky方向上的跳跃 (σx)

addhopping!(BN, [1, -1, 0], 1, 2, λ/2) # kx-ky方向上的跳跃 (σy)
addhopping!(BN, [-1, 1, 0], 1, 2, -λ/2) # 反向跳跃 (σy)
addhopping!(BN, [1, -1, 0], 2, 1, -λ/2) # kx-ky方向上的跳跃 (σy)
addhopping!(BN, [-1, 1, 0], 2, 1, λ/2) # 反向跳跃 (σy)

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


kpath = zeros(Float64, 3, 5)
kpath[:, 1] = [0.5, 0.5, 0.0]  # M
kpath[:, 2] = [0.5, 0.0, 0.0]  # X
kpath[:, 3] = [0.0, 0.0, 0.0]  # Γ
kpath[:, 4] = [0.0, 0.5, 0.0]  # Y
kpath[:, 5] = [0.5, 0.5, 0.0]  # Γ
kdist, egvals = HopTB.BandStructure.getbs(BN, kpath, 50, connect_end_points=true)

fig = figure("Band structure monolayer hBN",figsize=(10,20))

max_val=findmax(egvals[1,:])[1]
egvals=egvals.-max_val

plot(kdist,egvals[1,:],linestyle="-")
plot(kdist,egvals[2,:],linestyle="--")
PyPlot.axhline(0.0,color="black",linestyle="-.")
PyPlot.show()