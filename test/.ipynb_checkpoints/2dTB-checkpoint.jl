using HopTB
using PyPlot


sethopping!(BN, [1, 0, 0], 1, 1, -t)  # 跳跃到右侧相邻格点
sethopping!(BN, [-1, 0, 0], 1, 1, -t) # 跳跃到左侧相邻格点
sethopping!(BN, [0, 1, 0], 1, 1, -t)  # 跳跃到上方相邻格点
sethopping!(BN, [0, -1, 0], 1, 1, -t) # 跳跃到下方相邻格点
# 跳跃到对角线方向 (1, 1)
sethopping!(BN, [1, 1, 0], 1, 2, λ/2, σx) 
sethopping!(BN, [-1, -1, 0], 2, 1, λ/2, σx)

# 跳跃到对角线方向 (1, -1)
sethopping!(BN, [1, -1, 0], 1, 2, λ/2, σy)
sethopping!(BN, [-1, 1, 0], 2, 1, λ/2, σy)
# 对于 J1 (cos(kx) - cos(ky)) 项
sethopping!(BN, [1, 0, 0], 1, 1, J1, σz)   # cos(kx) 部分
sethopping!(BN, [0, 1, 0], 1, 1, -J1, σz)  # -cos(ky) 部分

# 对于 J2 sin(kx)sin(ky) 项
sethopping!(BN, [1, 1, 0], 1, 1, J2, σz)    # 对角线方向 (1, 1)
sethopping!(BN, [-1, -1, 0], 1, 1, J2, σz)  # 对角线方向 (-1, -1)
