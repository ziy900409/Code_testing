import numpy as np
from scipy.spatial.transform import Rotation
import imufusion
import matplotlib.pyplot as pyplot
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from mpl_toolkits.mplot3d import Axes3D


# Load data from CSV
data = np.genfromtxt(r"E:\Hsin\git\TWU\KaiCode\golf.csv", delimiter=",", skip_header=1)
timestamp = data[:, 0]
quat = data[:, 10:14]
accelerometer = data[:, 7:10]

# Iterate through the data and rotate each accelerometer reading
global_accelerometer = []
for q, acc in zip(quat, accelerometer):
    # Create a Rotation object from the quaternion
    rotation = Rotation.from_quat(q)

    # Rotate the accelerometer reading to global coordinates
    global_acc = rotation.apply(acc)

    # Append the global accelerometer reading to the list
    global_accelerometer.append(global_acc)

# Convert the global accelerometer data to a NumPy array
global_accelerometer = np.array(global_accelerometer)

lowcut = 10  # 低通滤波的截止频率
order = 4  # 滤波器阶数

# 计算正常化截止频率
nyq = 0.5 * 120.0  # 假设采样率为1000 Hz
low = lowcut / nyq

# 创建Butterworth低通滤波器
b, a = butter(order, low, btype='low')

# 应用滤波器到陀螺仪数据
global_accelerometer = lfilter(b, a, global_accelerometer, axis=0)
accx = global_accelerometer[250:,0]
accy = global_accelerometer[250:,1]
accz = global_accelerometer[250:,2]
global_accelerometer = np.vstack((accx, accy, accz)).T
# 畫圖
# plt.plot(accx)
# plt.plot(accy)
# plt.plot(accz)

def double_integration(global_accelerometer, frequency):
    num_samples = len(global_accelerometer)
    time_intervals = 1 / frequency * np.arange(num_samples)
    velocity_data = np.zeros((num_samples, 3))
    position_data = np.zeros((num_samples, 3))

    for i in range(1, num_samples):
        delta_t = time_intervals[i] - time_intervals[i - 1]
        velocity_data[i] = velocity_data[i - 1] + global_accelerometer[i] * delta_t
        position_data[i] = position_data[i - 1] + velocity_data[i] * delta_t

    return position_data

frequency = 120
num_samples = 1631
position_data = double_integration(global_accelerometer, frequency)

x_positions = position_data[:, 0] 
y_positions = position_data[:, 1]
z_positions = position_data[:, 2]

# 畫圖
plt.plot(position_data[:, 0], label='X Position')
plt.plot(position_data[:, 1], label='Y Position')
plt.plot(position_data[:, 2], label='Z Position')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_positions, y_positions, z_positions)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

plt.show()