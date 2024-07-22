import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 确保使用正确的 Matplotlib 后端
import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'TkAgg' Qt5Agg

# 如果在 Spyder 中，确保绘图窗口是独立的
# 在 Spyder 的 IPython 控制台中运行以下命令：
# %matplotlib qt

# 读取数据
raw_data = pd.read_csv(r"C:\Users\h7058\Downloads\d_1.csv")

# 数据参数
num_points = 63
num_frames = 1520

# 创建一个空的数据数组
data = np.zeros([num_points, num_frames, 3])

# 填充数据数组
for idx in range(num_points):
    data[idx, :, 0] = raw_data.iloc[1:, idx*3 + 3].values
    data[idx, :, 1] = raw_data.iloc[1:, idx*3 + 4].values
    data[idx, :, 2] = raw_data.iloc[1:, idx*3 + 5].values

# 创建一个3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化一个空的散点图
scatter = ax.scatter([], [], [], c='r', marker='o')

# 初始化不同颜色的点
highlight_point = ax.scatter([], [], [], c='b', marker='x')

# 设置轴的范围，根据你的数据实际范围调整
ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
ax.set_zlim(np.min(data[:, :, 2]), np.max(data[:, :, 2]))

# 更新函数，用于动画
def update(frame):
    # 更新散点图的数据
    scatter._offsets3d = (data[:, frame, 0], data[:, frame, 1], data[:, frame, 2])
    # 设置高亮点的坐标（例如在每一帧中的第一个点）
    highlight_x = data[-1, frame, 0]
    highlight_y = data[-1, frame, 1]
    highlight_z = data[-1, frame, 2]
    highlight_point._offsets3d = ([highlight_x], [highlight_y], [highlight_z])
    
    return scatter, highlight_point

# 创建动画
ani = FuncAnimation(fig, update, frames=num_frames, interval=20, blit=False)


# 显示图形
plt.show()
