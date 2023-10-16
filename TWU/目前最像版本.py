import imufusion
import matplotlib.pyplot as plt
import numpy
import sys
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter

# Import sensor data
data = numpy.genfromtxt(
    "E:\git\Code_testing\TWU\data\golf.csv", delimiter=",", skip_header=1
)

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]

# Plot sensor data
_, axes = plt.subplots(nrows=3, sharex=True)

axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="X")
axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Y")
axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Z")
axes[0].set_title("Gyroscope")
axes[0].set_ylabel("Degrees/s")
axes[0].grid()
axes[0].legend()

axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="X")
axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Y")
axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Z")
axes[1].set_title("Accelerometer")
axes[1].set_ylabel("g")
axes[1].grid()
axes[1].legend()

# Process sensor data
ahrs = imufusion.Ahrs()
euler = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    ahrs.update_no_magnetometer(
        gyroscope[index], accelerometer[index], 1 / 100
    )  # 100 Hz sample rate
    euler[index] = ahrs.quaternion.to_euler()

# Plot Euler angles
axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[2].set_title("Euler angles")
axes[2].set_xlabel("Seconds")
axes[2].set_ylabel("Degrees")
axes[2].grid()
axes[2].legend()

plt.show(block="no_block" not in sys.argv)  # don't block when script run by CI

quaternions = []
for euler_angles in euler:
    r = R.from_euler("xyz", euler_angles, degrees=True)
    quaternion = r.as_quat()
    quaternions.append(quaternion)

# Convert the list of quaternions to a numpy array
quaternions = numpy.array(quaternions)

# 先無條件相信這個q_明天我收data直接用x sens提供的q
rm = numpy.empty((len(timestamp), 3, 3))

for i in range(len(rm)):
    component = numpy.array(
        [
            (2 * quaternions[i, 0] ** 2 + 2 * quaternions[i, 1] ** 2 - 1),
            (
                2 * quaternions[i, 1] * quaternions[i, 2]
                - 2 * quaternions[i, 0] * quaternions[i, 3]
            ),
            (
                2 * quaternions[i, 0] * quaternions[i, 2]
                + 2 * quaternions[i, 1] * quaternions[i, 3]
            ),
            (
                2 * quaternions[i, 1] * quaternions[i, 2]
                + 2 * quaternions[i, 0] * quaternions[i, 3]
            ),
            (2 * quaternions[i, 0] ** 2 + 2 * quaternions[i, 2] ** 2 - 1),
            (
                2 * quaternions[i, 2] * quaternions[i, 3]
                - 2 * quaternions[i, 0] * quaternions[i, 1]
            ),
            (
                2 * quaternions[i, 1] * quaternions[i, 3]
                - 2 * quaternions[i, 0] * quaternions[i, 2]
            ),
            (
                2 * quaternions[i, 2] * quaternions[i, 3]
                + 2 * quaternions[i, 0] * quaternions[i, 1]
            ),
            (2 * quaternions[i, 2] ** 2 + 2 * quaternions[i, 3] ** 2 - 1),
        ]
    )
    rm[i] = component.reshape((3, 3))


# Iterate through the data and rotate each accelerometer reading
global_accelerometer = []
for q, acc in zip(quaternions, accelerometer):
    # Create a Rotation object from the quaternion
    rotation = R.from_quat(q)

    # Rotate the accelerometer reading to global coordinates
    global_acc = rotation.apply(acc)

    # Append the global accelerometer reading to the list
    global_accelerometer.append(global_acc)

# Convert the global accelerometer data to a NumPy array
global_accelerometer = numpy.array(global_accelerometer)

lowcut = 10  # 低通滤波的截止频率
order = 4  # 滤波器阶数

# 计算正常化截止频率
nyq = 0.5 * 120.0  # 假设采样率为1000 Hz
low = lowcut / nyq

# 创建Butterworth低通滤波器
b, a = butter(order, low, btype="low")

# 应用滤波器到陀螺仪数据
global_accelerometer = lfilter(b, a, global_accelerometer, axis=0)
accx = global_accelerometer[250:, 0]
accy = global_accelerometer[250:, 1]
accz = global_accelerometer[250:, 2]
global_accelerometer = numpy.vstack((accx, accy, accz)).T
# 畫圖
plt.plot(accx)
plt.plot(accy)
plt.plot(accz)
plt.title("global acc")


def double_integration(global_accelerometer, frequency):
    num_samples = len(global_accelerometer)
    time_intervals = 1 / frequency * numpy.arange(num_samples)
    velocity_data = numpy.zeros((num_samples, 3))
    position_data = numpy.zeros((num_samples, 3))

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
plt.plot(position_data[:, 0], label="X Position")
plt.plot(position_data[:, 1], label="Y Position")
plt.plot(position_data[:, 2], label="Z Position")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(x_positions, y_positions, z_positions)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
plt.title("global position")

plt.show()
