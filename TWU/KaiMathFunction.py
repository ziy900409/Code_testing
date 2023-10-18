# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 19:09:26 2023

@author: Hsin Yang, 16.Oct.2023
"""
# %% import package
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

# %% define Math function


def HessianProduct(q_1, q_2, shape="1D"):
    """
    calculate the quaternion product of quaternion q_1 and q_2

    Returns:
    Prod (numpy.array): Quaternion product of q_1 and q_2.
    """
    # q_1 = np.array(q)
    # q_2 = np.array([0, gyroscope[0], gyroscope[1], gyroscope[2]])
    # shape = "1D"
    if shape == "1D":
        q_1 = np.array(q_1)
        q_2 = np.array(q_2)
        Prod = np.zeros_like(q_1)
        Prod[0] = q_1[0] * q_2[0] - q_1[1] * q_2[1] - q_1[2] * q_2[2] - q_1[3] * q_2[3]
        Prod[1] = q_1[0] * q_2[1] + q_1[1] * q_2[0] + q_1[2] * q_2[3] - q_1[3] * q_2[2]
        Prod[2] = q_1[0] * q_2[2] - q_1[1] * q_2[3] + q_1[2] * q_2[0] + q_1[3] * q_2[1]
        Prod[3] = q_1[0] * q_2[3] + q_1[1] * q_2[2] - q_1[2] * q_2[1] + q_1[3] * q_2[0]
    elif shape == "2D":
        Prod = np.zeros_like(q_1)
        Prod[:, 0] = (
            q_1[:, 0] * q_2[:, 0]
            - q_1[:, 1] * q_2[:, 1]
            - q_1[:, 2] * q_2[:, 2]
            - q_1[:, 3] * q_2[:, 3]
        )
        Prod[:, 1] = (
            q_1[:, 0] * q_2[:, 1]
            + q_1[:, 1] * q_2[:, 0]
            + q_1[:, 2] * q_2[:, 3]
            - q_1[:, 3] * q_2[:, 2]
        )
        Prod[:, 2] = (
            q_1[:, 0] * q_2[:, 2]
            - q_1[:, 1] * q_2[:, 3]
            + q_1[:, 2] * q_2[:, 0]
            + q_1[:, 3] * q_2[:, 1]
        )
        Prod[:, 3] = (
            q_1[:, 0] * q_2[:, 3]
            + q_1[:, 1] * q_2[:, 2]
            - q_1[:, 2] * q_2[:, 1]
            + q_1[:, 3] * q_2[:, 0]
        )

    return Prod


def quatern2rotMat(q, shape="1D"):
    """
    Converts a quaternion orientation to a rotation matrix.

    Parameters:
    q (numpy.array): Input quaternion in the form [w, x, y, z].

    Returns:
    R (numpy.array): Rotation matrix.
    """
    if shape == "1D":
        R = np.zeros((3, 3))
        R[0, 0] = 2 * (q[0] ** 2) - 1 + 2 * (q[1] ** 2)
        R[0, 1] = 2 * (q[1] * q[2] + q[0] * q[3])
        R[0, 2] = 2 * (q[1] * q[3] - q[0] * q[2])
        R[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
        R[1, 1] = 2 * (q[0] ** 2 - 1 + 2 * (q[2] ** 2))
        R[1, 2] = 2 * (q[2] * q[3] + q[0] * q[1])
        R[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
        R[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
        R[2, 2] = 2 * (q[0] ** 2 - 1 + 2 * (q[3] ** 2))
    elif shape == "2D":
        R = np.zeros((len(q), 3, 3))
        R[:, 0, 0] = 2 * (q[:, 0] ** 2) - 1 + 2 * (q[:, 1] ** 2)
        R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        R[:, 1, 1] = 2 * (q[:, 0] ** 2 - 1 + 2 * (q[:, 2] ** 2))
        R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
        R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
        R[:, 2, 2] = 2 * (q[:, 0] ** 2 - 1 + 2 * (q[:, 3] ** 2))

    return R


class Quaternion:
    """
    還需要再修改
    """

    def __init__(self, real, imag_i, imag_j, imag_k):
        self.real = real
        self.imag_i = imag_i
        self.imag_j = imag_j
        self.imag_k = imag_k

    def conjugate(self):
        self.imag_i = -self.imag_i
        self.imag_j = -self.imag_j
        self.imag_k = -self.imag_k
        return np.array([self.real, self.imag_i, self.imag_j, self.imag_k])

    def __str__(self):
        return "[{}  {}i  {}j  {}k]".format(
            self.real, self.imag_i, self.imag_j, self.imag_k
        )

    @staticmethod
    def multiply(q1, q2):
        real_part = (
            q1.real * q2.real
            - q1.imag_i * q2.imag_i
            - q1.imag_j * q2.imag_j
            - q1.imag_k * q2.imag_k
        )
        imag_i_part = (
            q1.real * q2.imag_i
            + q1.imag_i * q2.real
            + q1.imag_j * q2.imag_k
            - q1.imag_k * q2.imag_j
        )
        imag_j_part = (
            q1.real * q2.imag_j
            - q1.imag_i * q2.imag_k
            + q1.imag_j * q2.real
            + q1.imag_k * q2.imag_i
        )
        imag_k_part = (
            q1.real * q2.imag_k
            + q1.imag_i * q2.imag_j
            - q1.imag_j * q2.imag_i
            + q1.imag_k * q2.real
        )
        return Quaternion(real_part, imag_i_part, imag_j_part, imag_k_part)

    @staticmethod
    def conjugate(quaternion):
        w, x, y, z = quaternion
        return np.array([w, -x, -y, -z])

    @staticmethod
    def ToRotMat(q):
        """
        Converts a quaternion orientation to a rotation matrix.

        Parameters:
        q (numpy.array): Input quaternion in the form [w, x, y, z].

        Returns:
        R (numpy.array): Rotation matrix.
        """
        R = np.zeros((len(q), 3, 3))
        R[:, 0, 0] = 2 * (q[:, 0] ** 2) - 1 + 2 * (q[:, 1] ** 2)
        R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        R[:, 1, 1] = 2 * (q[:, 0] ** 2 - 1 + 2 * (q[:, 2] ** 2))
        R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
        R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
        R[:, 2, 2] = 2 * (q[:, 0] ** 2 - 1 + 2 * (q[:, 3] ** 2))

        return R


# %%

"""
# 創建兩個四元素
quat1 = Quaternion(1, 2, 3, 4)
quat2 = Quaternion(5, 6, 7, 8)
# quat2_conJ = Quaternion.conjugate()
# 計算四元素的乘法
result = Quaternion.multiply(quat1, quat2)

# 輸出結果
print("四元素1:", quat1)
print("四元素2:", quat2)
print("四元素2 conjugate:", quat2.conjugate())
print("四元素2 conjugate:", quat2)
print("四元素乘法結果:", result)
"""

# %%
