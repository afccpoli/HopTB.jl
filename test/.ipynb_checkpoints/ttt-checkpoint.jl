using PyPlot
plt.ion()
x = 1:10
y = rand(10)

plot(x, y, label="随机数据")
xlabel("X 轴")
ylabel("Y 轴")
title("示例图表")
legend()
