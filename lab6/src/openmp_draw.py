import matplotlib.pyplot as plt

plt.style.use('seaborn')

# 数据
matrix_sizes = [16, 64, 128, 256, 512, 1024]
gflops = [0.010436, 0.091691, 1.035119, 1.190422, 0.787312, 0.455611]

# 绘制曲线图
plt.plot(matrix_sizes, gflops, marker='o', linestyle='-')
plt.title('OpenMP\'s Gflops')
plt.xlabel('Matrix Size')
plt.ylabel('GFLOPS')
plt.grid(True)

# 显示图表
plt.show()
