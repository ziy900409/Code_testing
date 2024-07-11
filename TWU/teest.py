import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


raw_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\d_1.csv")


# 创建示例数据
# 假设我们有10个时间点，每个时间点上有5个标记点
num_frames = 1520
num_points = 62

# 生成随机数据作为示例 (每个点在 (x, y, z) 轴上的坐标)
data = np.random.rand(num_frames, num_points, 3)

data = np.zeros([62, 1520, 3])
for idx in range(num_points):
    data[idx, :, 0] = raw_data.iloc[1:, idx*3 + 3].values
    data[idx, :, 1] = raw_data.iloc[1:, idx*3 + 4].values
    data[idx, :, 2] = raw_data.iloc[1:, idx*3 + 5].values

# 创建一个3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化一个空的散点图
scatter = ax.scatter([], [], [], c='r', marker='o')

# 设置轴的范围，根据你的数据实际范围调整
ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
ax.set_zlim(np.min(data[:, :, 2]), np.max(data[:, :, 2]))

# 更新函数，用于动画
def update(frame):
    # 更新散点图的数据
    scatter._offsets3d = (data[:, frame, 0], data[:, frame, 1], data[:, frame, 2])
    return scatter,

# 创建动画
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# 显示图形
plt.show()
