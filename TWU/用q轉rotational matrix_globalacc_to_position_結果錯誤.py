import imufusion
import matplotlib.pyplot as pyplot
import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from mpl_toolkits.mplot3d import Axes3D


data = np.genfromtxt("E:\Hsin\git\TWU\KaiCode\golf.csv", delimiter=",", skip_header=1)
timestamp = data[:, 0]
q = data[:, 10:14]
accelerometer = data[:, 7:10]
freeaccelerometer = data[:, 4:7]
gyroscope = data[:, 1:4]


# 指定滤波器参数
lowcut = 5  # 低通滤波的截止频率
order = 4  # 滤波器阶数

# 计算正常化截止频率
nyq = 0.5 * 20.0  # 假设采样率为1000 Hz
low = lowcut / nyq

# 创建Butterworth低通滤波器
b, a = butter(order, low, btype='low')

# 应用滤波器到陀螺仪数据
gyroscope = lfilter(b, a, gyroscope, axis=0)
accelerometer = lfilter(b, a, accelerometer, axis=0)
freeaccelerometer = lfilter(b, a, freeaccelerometer, axis=0)


rm = np.empty((len(timestamp), 3, 3))

for i in range(len(timestamp)):
    r11 = 2 * q[i, 0]**2 + 2 * q[i, 1]**2 - 1
    r12 = 2 * q[i, 1] * q[i, 2] - 2 * q[i, 0] * q[i, 3]
    r13 = 2 * q[i, 0] * q[i, 2] + 2 * q[i, 1] * q[i, 3]
    r21 = 2 * q[i, 1] * q[i, 2] + 2 * q[i, 0] * q[i, 3]
    r22 = 2 * q[i, 0]**2 + 2 * q[i, 2]**2 - 1
    r23 = 2 * q[i, 2] * q[i, 3] - 2 * q[i, 0] * q[i, 1]
    r31 = 2 * q[i, 1] * q[i, 3] - 2 * q[i, 0] * q[i, 2]
    r32 = 2 * q[i, 2] * q[i, 3] + 2 * q[i, 0] * q[i, 1]
    r33 = 2 * q[i, 0]**2 + 2 * q[i, 3]**2 - 1
    rm[i] = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    
tm = np.empty((len(timestamp), 3, 3))

for i in range(len(timestamp)):
    t11 = 2 * q[i, 0]**2 + 2 * q[i, 1]**2 - 1
    t12 = 2 * q[i, 1] * q[i, 2] - 2 * q[i, 0] * q[i, 3]
    t13 = 2 * q[i, 0] * q[i, 2] + 2 * q[i, 1] * q[i, 3]
    t21 = 2 * q[i, 1] * q[i, 2] + 2 * q[i, 0] * q[i, 3]
    t22 = 2 * q[i, 0]**2 + 2 * q[i, 2]**2 - 1
    t23 = 2 * q[i, 2] * q[i, 3] - 2 * q[i, 0] * q[i, 1]
    t31 = 2 * q[i, 1] * q[i, 3] - 2 * q[i, 0] * q[i, 2]
    t32 = 2 * q[i, 2] * q[i, 3] + 2 * q[i, 0] * q[i, 1]
    t33 = 2 * q[i, 0]**2 + 2 * q[i, 3]**2 - 1
    tm[i] = [[r11, r21, r31], [r12, r22, r32], [r13, r23, r33]]
    

    

num_frames = accelerometer.shape[0]
results = []

# 使用 for 迴圈處理每個 frame 並計算點積
for i in range(num_frames):
    result = np.dot(rm[i], accelerometer[i])
    results.append(result)
    
global_acc1 = np.array(results)

accx = (global_acc1[:, 0])
accy = (global_acc1[:, 1])
accz = (global_acc1[:, 2])

glocal_acc = np.vstack((accx, accy,accz)).T

plt.plot(glocal_acc[:, 0])
plt.plot(glocal_acc[:, 1])
plt.plot(glocal_acc[:, 2])


def double_integration(global_acc, frequency):
    num_samples = len(global_acc)
    time_intervals = 1 / frequency * np.arange(num_samples)
    velocity_data = np.zeros((num_samples, 3))
    position_data = np.zeros((num_samples, 3))

    for i in range(1, num_samples):
        delta_t = time_intervals[i] - time_intervals[i - 1]
        velocity_data[i] = velocity_data[i - 1] + global_acc[i] * delta_t
        position_data[i] = position_data[i - 1] + velocity_data[i] * delta_t

    return position_data

frequency = 20
num_samples = 192
position_data = double_integration(glocal_acc, frequency)

x_positions = position_data[:, 0] 
y_positions = position_data[:, 1]
z_positions = position_data[:, 2]

plt.plot(position_data[:, 0], label='X Position')
plt.plot(position_data[:, 1], label='Y Position')
plt.plot(position_data[:, 2], label='Z Position')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_positions, x_positions, z_positions, label='Position')
ax.set_ylabel('Y Position (units)')
ax.set_xlabel('X Position (units)')
ax.set_zlabel('Z Position (units)')
ax.set_title('3D Position vs Time')

ax.set_box_aspect([1, 1, 1])

plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y_positions, x_positions, label='Position')
ax.set_ylabel('Y Position (units)')
ax.set_xlabel('X Position (units)')
ax.set_title('2D Position vs Time')
plt.legend()
plt.show()
