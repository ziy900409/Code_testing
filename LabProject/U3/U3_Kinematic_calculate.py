# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 18:16:31 2023

計算順序 :
    0. 裁切data : EMG, motion data
    1. 運動學計算 :
        1.1. Tpose 計算回所有的 R.Elbow.Med位置
            1.1. func
        1.2. Tpose 定義人體自然姿勢，定義offset_Rot
        1.3. 計算手肘、手腕之尤拉角
            1.3.0. 定義近端與遠端的旋轉矩陣
            1.3.1. 計算支段的旋轉矩陣
            1.3.2. 將每個frame的旋轉矩陣轉換為毆拉參數
            1.3.3. 使用毆拉參數計算角速度與角加速度
            1.3.4. 再將毆拉參數轉換為旋轉矩陣，在將旋轉矩陣轉換成歐拉角
        1.4. 匯出.c3d or .trc
    2. 找尋所最大關節角加速度位置




@author: Hsin.YH.Yang
"""
# %% import package
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\U3")
# import U3_Kinematic_function as func

# import os
import numpy as np
import pandas as pd
# from scipy.signal import find_peaks
# import gc
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import math

# %% 定義支段座標系之旋轉矩陣
## 定義上臂座標系
def DefCoordArm(R_Shoulder, R_Elbow_Lat, R_Elbow_Med):
    """
    i-axis : R.Shoulder - (R.Elbow.Lat + R.Elbow.Med)/2
    j-axis : i cross k
    k-axis : R.Elbow.Lat - R.Elbow.Med
    
    R = [[ix, iy, iz],
         [jx, jy, jz],
         [kx, ky, kz]]

    Parameters
    ----------
    R_Shoulder : numpy.array
        DESCRIPTION.
    R_Elbow_Lat : numpy.array
        DESCRIPTION.
    R_Elbow_Med : numpy.array
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 將 input 轉成 np.array
    R_Shoulder = np.array(R_Shoulder)
    R_Elbow_Lat = np.array(R_Elbow_Lat)
    R_Elbow_Med = np.array(R_Elbow_Med)
    # 定義座標軸方向
    j_vector = R_Shoulder - (R_Elbow_Lat + R_Elbow_Med)/2
    i_vector = np.cross(j_vector, (R_Elbow_Lat - R_Elbow_Med))
    k_vector = np.cross(i_vector, j_vector)
    # 定義單位方向
    j_axis = j_vector / np.linalg.norm(j_vector)
    i_axis = np.cross(j_axis, i_vector) / np.linalg.norm(np.cross(j_axis, i_vector))
    k_axis = k_vector / np.linalg.norm(k_vector)

    RotMatrix = np.array([i_axis, j_axis, k_axis])
    return RotMatrix
## 定義小臂座標系
def DefCoordForearm(R_Elbow_Lat, R_Elbow_Med, R_Wrist_Una, R_Wrist_Rad):
    """
    針對右手
    i-axis : (R_Elbow_Lat + R_Elbow_Med)/2 - (R_Wrist_Una + R_Wrist_Rad)/2
    j-axis : x cross z
    k-axis : R_Wrist_Rad - R_Wrist_Una
    
    R = [[ix, iy, iz],
         [jx, jy, jz],
         [kx, ky, kz]]

    Parameters
    ----------
    R_Elbow_Lat : TYPE
        DESCRIPTION.
    R_Elbow_Med : TYPE
        DESCRIPTION.
    R_Wrist_Una : TYPE
        DESCRIPTION.
    R_Wrist_Rad : TYPE
        DESCRIPTION.

    Returns
    -------
    RotMatrix : TYPE
        DESCRIPTION.

    """
    # 將 input 轉成 np.array
    R_Elbow_Lat = np.array(R_Elbow_Lat)
    R_Elbow_Med = np.array(R_Elbow_Med)
    R_Wrist_Una = np.array(R_Wrist_Una)
    R_Wrist_Rad = np.array(R_Wrist_Rad)
    # 定義座標軸方向
    # j_vector = (R_Elbow_Lat + R_Elbow_Med)/2 - (R_Wrist_Una + R_Wrist_Rad)/2
    j_vector = (R_Elbow_Lat + R_Elbow_Med)/2 - R_Wrist_Una 
    i_vector = np.cross(j_vector, (R_Wrist_Rad - R_Wrist_Una))
    k_vector = np.cross(i_vector, j_vector)
    
    j_axis = j_vector / np.linalg.norm(j_vector)
    i_axis = i_vector / np.linalg.norm(i_vector)
    k_axis = k_vector / np.linalg.norm(k_vector)
    
    RotMatrix = np.array([i_axis, j_axis, k_axis])
    return RotMatrix

## 定義手掌座標系
def DefCoordHand(R_Wrist_Una, R_Wrist_Rad, R_M_Finger1):
    """
    i-axis : (R_Wrist_Una + R_Wrist_Rad)/2 - R_M_Finger1
    j-axis : x cross z
    k-axis : R_Wrist_Rad - R_Wrist_Una
    
    R = [[ix, iy, iz],
         [jx, jy, jz],
         [kx, ky, kz]]
    
    Parameters
    ----------
    R_Shoulder : TYPE
        DESCRIPTION.
    R_Elbow_Lat : TYPE
        DESCRIPTION.
    R_Elbow_Med : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 將 input 轉成 np.array
    R_Wrist_Una = np.array(R_Wrist_Una)
    R_Wrist_Rad = np.array(R_Wrist_Rad)
    R_M_Finger1 = np.array(R_M_Finger1)
    # 定義座標軸方向
    j_vector = (R_Wrist_Una + R_Wrist_Rad)/2 - R_M_Finger1
    
    # i_vector = np.cross(j_vector, (R_Wrist_Rad - R_Wrist_Una))
    i_vector = np.cross(j_vector, (R_Wrist_Rad - R_M_Finger1))
    k_vector = np.cross(i_vector, j_vector)
    
    
    j_axis = j_vector / np.linalg.norm(j_vector)
    i_axis = i_vector / np.linalg.norm(i_vector)
    k_axis = k_vector / np.linalg.norm(k_vector)
    
    RotMatrix = np.array([i_axis, j_axis, k_axis])
    return RotMatrix
# %% 計算兩坐標系之間的旋轉矩陣
def joint_angle_rot(RotP, RotD, OffsetRotP=None, OffsetRotD=None):
    """
    Parameters
    ----------
    RotP : numpy.array
        proximal segment rotation matrix (3 X 3 X n frames).
    RotD : numpy.array
        distal segment rotation matrix (3 X 3 X n frames).
    OffsetRotP : TYPE
        joint angle offset distal rotation matrix (3 X 3).
    OffsetRotD : TYPE
        joint angle offset proximal rotation matrix (3 X 3).

    Returns
    -------
    Rot : numpy.array
        joint angles rotation matrix (3 X 3 X n frames).
    
    """
    n_frames = RotD.shape[2]  # 獲取旋轉矩陣的第三個維度，即幀數量
    Rot = np.zeros((3, 3, n_frames))  # 創建一個空的旋轉矩陣，維度為 3x3xN(幀數量)

    if OffsetRotD is None and OffsetRotP is None:
        # 如果未提供偏移矩陣，則直接計算 RotP 與 RotD 的轉置矩陣之積
        for i in range(n_frames):
            # Rot[:, :, i] = np.transpose(np.dot(RotP[:, :, i], RotD[:, :, i].T))
            Rot[:, :, i] = np.dot(RotP[:, :, i], RotD[:, :, i].T)
            
    elif OffsetRotD is not None and OffsetRotP is None:
        # 如果提供了 RotD 的偏移矩陣 OffsetRotD，則計算 OffsetRotD 與 RotP 轉置矩陣之積，再與 RotD 轉置矩陣相乘
        for i in range(n_frames):
            Rot[:, :, i] = np.dot(np.dot(OffsetRotD.T, RotP[:, :, i]), RotD[:, :, i].T)
            
    elif OffsetRotD is not None and OffsetRotP is not None:
        # 如果同時提供了 RotD 和 RotP 的偏移矩陣 OffsetRotD 和 OffsetRotP，則計算 OffsetRotP 與 OffsetRotD 轉置矩陣之積，再與 Rp 轉置矩陣和 Rd 轉置矩陣相乘
        OffsetR = np.dot(OffsetRotP, OffsetRotD.T)
        for i in range(n_frames):
            Rot[:, :, i] = np.dot(np.dot(OffsetR.T, RotP[:, :, i]), RotD[:, :, i].T)
            
    return Rot
# %% 將旋轉矩陣轉為尤拉角

def Rot2EulerAngle(Rot, sequence, Unit='deg'):
    """

    Parameters
    ----------
    Rot : TYPE
        DESCRIPTION.
    sequence : TYPE
        DESCRIPTION.
    Unit : TYPE, optional
        DESCRIPTION. The default is 'deg'.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.

    """
    if Unit == 'rad':
        Rot = np.radians(Rot)
        
    theta = np.empty(shape=(np.shape(Rot)[-1], 3))

    for i in range(Rot.shape[2]):
        if sequence == 'xyz':
            # % x axis
            theta[i, 0] = np.degrees(np.arctan2(-Rot[1, 2, i], Rot[2, 2, i]))
            # % y axis
            theta[i, 1] = np.degrees(np.arctan2(Rot[0, 2, i],((Rot[1, 2, i])**2+(Rot[2, 2, i])**2)**0.5))
            # % z axis
            theta[i, 2] = np.degrees(np.arctan2(-Rot[0, 1, i],Rot[0, 0, i]))        
        
        elif sequence == 'xzy':        
            theta[i, 0] = np.degrees(np.arctan2(Rot[2, 1, i],Rot[1, 1, i]))
            theta[i, 1] = np.degrees(np.arctan2(Rot[0, 2, i],Rot[0, 0, i]))
            theta[i, 2] = np.degrees(np.arctan2((-Rot[0, 1, i]),((Rot[1, 1, i])**2+(Rot[2, 1, i])**2)**0.5))      
            
        elif sequence == 'yxz':        
            theta[i, 0] = np.degrees(np.arctan2((-Rot[1, 2, i]),((Rot[0, 2, i])**2+(Rot[1, 2, i])**2)**0.5))
            theta[i, 1] = np.degrees(np.arctan2(Rot[0, 2, i],Rot[1, 2, i]))
            theta[i, 2] = np.degrees(np.arctan2(Rot[1, 0, i],Rot[1, 1, i]))
            
        elif sequence == 'yzx':        
            theta[i, 0] = np.degrees(np.arctan2(-Rot[1, 2, i],Rot[1, 1, i]))
            theta[i, 1] = np.degrees(np.arctan2(-Rot[2, 0, i],Rot[0, 0, i]))
            theta[i, 2] = np.degrees(np.arctan2((Rot[1, 0, i]),((Rot[2, 0, i])**2+(Rot[0, 0, i])**2)**0.5))
            
        elif sequence == 'zxy':        
            theta[i, 0] = np.degrees(np.arctan2((Rot[2, 1, i]),((Rot[0, 1, i])**2+(Rot[1, 1, i])**2)**0.5))
            theta[i, 1] = np.degrees(np.arctan2(-Rot[2, 0, i],Rot[2, 2, i]))
            theta[i, 2] = np.degrees(np.arctan2(-Rot[0, 1, i],Rot[1, 1, i]))
            
        elif sequence == 'zyx':        
            theta[i, 0] = np.degrees(np.arctan2(Rot[2, 1, i],Rot[2, 2, i]))
            theta[i, 1] = np.degrees(np.arctan2((-Rot[2, 0, i]),(((Rot[1, 0, i])**2+(Rot[0, 0, i])**2)**0.5)))
            # theta[i, 1] = np.degrees(np.arctan2((-Rot[2, 0, i]),(((1-(Rot[0, 0, i])**2)**0.5))))
            theta[i, 2] = np.degrees(np.arctan2(Rot[1, 0, i],Rot[0, 0, i]))        
            
        elif sequence == 'xyx':
            theta[i, 0] = np.degrees(np.arctan2(Rot[1, 0, i],-Rot[2, 0, i]))
            theta[i, 1] = np.degreesnp.arctan2(((Rot[0, 1, i])**2+(Rot[0, 2, i])**2)**0.5,Rot[0, 0, i])
            theta[i, 2] = np.degrees(np.arctan2(Rot[0, 1, i],Rot[0, 2, i]))
            
        elif sequence == 'xzx':
            theta[i, 0] = np.degrees(np.arctan2(Rot[2, 0, i],Rot[1, 0, i]))
            theta[i, 1] = np.degreesnp.arctan2(((Rot[0, 1, i])**2+(Rot[0, 2, i])**2)**0.5,Rot[0, 0, i])
            theta[i, 2] = np.degrees(np.arctan2(Rot[0, 2, i],-Rot[0, 1, i]))
            
        elif sequence == 'yxy':
            theta[i, 0] = np.degrees(np.arctan2(Rot[0, 1, i],Rot[2, 1, i]))
            theta[i, 1] = np.degreesnp.arctan2(((Rot[0, 1, i])**2+(Rot[2, 1, i])**2)**0.5,Rot[1, 1, i])
            theta[i, 2] = np.degrees(np.arctan2(Rot[1, 0, i],-Rot[1, 2, i]))
            
        elif sequence == 'yzy':
            theta[i, 0] = np.degrees(np.arctan2(Rot[2, 1, i],-Rot[0, 1, i]))
            theta[i, 1] = np.degrees(np.arctan2(((Rot[0, 1, i])**2+(Rot[2, 1, i])**2)**0.5,Rot[1, 1, i]))
            theta[i, 2] = np.degrees(np.arctan2(Rot[1, 2, i],Rot[1, 0, i]))
            
        elif sequence == 'zxz':
            theta[i, 0] = np.degrees(np.arctan2(Rot[0, 2, i],-Rot[1, 2, i]))
            theta[i, 1] = np.degreesnp.arctan2(((Rot[0, 2, i])**2+(Rot[1, 2, i])**2)**0.5,Rot[3, 2, i])
            theta[i, 2] = np.degrees(np.arctan2(Rot[2, 0, i],Rot[2, 1, i]))
            
        elif sequence == 'zyz':
            theta[i, 0] = np.degrees(np.arctan2(Rot[1, 2, i],Rot[0, 2, i]))
            theta[i, 1] = np.degrees(np.arctan2(((Rot[0, 2, i])**2+(Rot[1, 2, i])**2)**0.5,Rot[2, 2, i]))
            theta[i, 2] = np.degrees(np.arctan2(Rot[2, 1, i],-Rot[2, 0, i]))        
    return theta


def Rot2EulerAngle_scipy(Rot, sequence, Unit='deg'):
    """
    

    Parameters
    ----------
    Rot : TYPE
        DESCRIPTION.
    sequence : TYPE
        DESCRIPTION.
    Unit : TYPE, optional
        DESCRIPTION. The default is 'deg'.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.

    """
    if Unit == 'rad':
        Rot = np.radians(Rot)
        
    theta = np.zeros((Rot.shape[2], 3))

    for i in range(Rot.shape[2]):
        if sequence in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx', 'xyx', 'xzx', 'yxy', 'yzy', 'zxz', 'zyz']:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(Rot[:, :, i])
            euler_angles = r.as_euler(sequence, degrees=True)
            theta[i] = euler_angles
        
    return theta

# %% 將旋轉矩陣轉成毆拉參數

# P, phi, n = Rot2EulerP(ElbowRot)

def Rot2EulerP(Rot):
    """
    將旋轉矩陣轉換成歐拉參數（Euler Parameters）
    
    1. 對於每一幀（第三個維度），將旋轉矩陣 R 的每一個片段轉置，以便將全局到本地的旋轉矩陣
    轉換為本地到全局的旋轉矩陣。

    2. 接下來，對於每一幀，計算歐拉參數 e0、e1、e2 和 e3。

    3. 如果 e0 等於 0，則根據旋轉矩陣的不同元素計算 e1、e2 和 e3。
    如果 e1 也等於 0，則進一步計算 e2，如果 e2 也等於 0，則 e3 設置為 1，否則計算 e3。 
    否則，如果 e0 不為 0，根據旋轉矩陣的不同元素計算 e1、e2 和 e3。.
    根據歐拉參數的值，計算旋轉角 phi 和旋轉軸 n。

    4. 如果 e0 等於 1，表示旋轉為零，將 phi 設置為 0，n 設置為 0。
    如果 e0 等於 0，表示旋轉為 180 度，將 phi 設置為 π，並根據 e1、e2 和 e3 設置 n。
    否則，計算 phi 為 2 * acos(e0)，並根據 e1、e2 和 e3 計算 n。
    將計算得到的歐拉參數 e0、e1、e2 和 e3 組合成一個矩陣 P。

    5. 最後，確保歐拉參數矩陣 P 的虛部部分為零，這是為了排除可能的複數結果。
    
    Parameters
    ----------
    Rot : TYPE
        輸入參數是一個旋轉矩陣 R（大小為 3x3xN，其中 N 是幀的數量）.


    Returns
    -------
    P : TYPE
        歐拉參數的矩陣.
    phi : TYPE
        旋轉角度的矩陣.
    n : TYPE
        旋轉軸的矩陣.
    
    """
    # Rot = ElbowRot
    # Initialize arrays for Euler parameters, rotation angles, and rotation axes
    # Rot = 
    e0 = np.zeros((Rot.shape[2], 1))
    e1 = np.zeros((Rot.shape[2], 1))
    e2 = np.zeros((Rot.shape[2], 1))
    e3 = np.zeros((Rot.shape[2], 1))
    phi = np.zeros((Rot.shape[2], 1))
    n = np.zeros((Rot.shape[2], 3))

    # Transpose the rotation matrices
    for ii in range(Rot.shape[2]):
        Rot[:, :, ii] = np.transpose(Rot[:, :, ii])

    # Calculate Euler parameters
    for i in range(Rot.shape[2]):
        e0[i, 0] = ((np.trace(Rot[:, :, i]) + 1) / 4) ** 0.5
        # 避免e0過小造成計算中出現NAN
        if e0[i, 0] == 0 or abs(e0[i, 0]) < 1e-10 or np.isnan(e0[i, 0]):
            '''
            需再確認
            or np.isnan(e0[i, 0])
            '''
            e0[i, 0] = 0
            e1[i, 0] = (-1 * (Rot[1, 1, i] + Rot[2, 2, i]) / 2) ** 0.5

            if e1[i, 0] == 0 or abs(e0[i, 0]) < 1e-10 or np.isnan(e0[i, 0]):
                e1[i, 0] = 0
                e2[i, 0] = ((1 - Rot[2, 2, i]) / 2) ** 0.5
                if e2[i, 0] == 0 or abs(e0[i, 0]) < 1e-10 or np.isnan(e0[i, 0]):
                    e2[i, 0] = 0
                    e3[i, 0] = 1
                else:
                    e3[i, 0] = Rot[2, 1, i] / (2 * e2[i, 0])
            else:
                e2[i, 0] = Rot[1, 0, i] / (2 * e1[i, 0])
                e3[i, 0] = Rot[2, 0, i] / (2 * e1[i, 0])
        else:
            e1[i, 0] = (Rot[2, 1, i] - Rot[1, 2, i]) / (4 * e0[i, 0])
            e2[i, 0] = (Rot[0, 2, i] - Rot[2, 0, i]) / (4 * e0[i, 0])
            e3[i, 0] = (Rot[1, 0, i] - Rot[0, 1, i]) / (4 * e0[i, 0])

        # Calculate rotation angle and rotation axis unit vectors
        if e0[i, 0] == 1:
            phi[i, 0] = 0
            n[i, 0:3] = 0
        elif e0[i, 0] == 0:
            phi[i, 0] = np.pi
            n[i, 0:3] = [e1[i, 0], e2[i, 0], e3[i, 0]]
        else:
            phi[i, 0] = 2 * np.arccos(e0[i, 0])
            n[i, 0] = e1[i, 0] / np.sin(phi[i, 0] / 2)
            n[i, 1] = e2[i, 0] / np.sin(phi[i, 0] / 2)
            n[i, 2] = e3[i, 0] / np.sin(phi[i, 0] / 2)

    P = np.hstack((e0, e1, e2, e3))
    P = np.real(P)
    # 計算角速度分量
    omega_x = 2 * (e0 * e1 + e2 * e3)
    omega_y = 2 * (e0 * e2 - e1 * e3)
    omega_z = 2 * (e0 * e3 + e1 * e2)
    
    omega = np.array([omega_x, omega_y, omega_z])
    
    return P, phi, n

# 獲取歐拉角，以度數形式
# from scipy.spatial.transform import Rotation

# 創建 Rotation 對象
# rotation_matrix = [[0.866, -0.5, 0],
#                    [0.5, 0.866, 0],
#                    [0, 0, 1]]

# rotations = Rotation.from_matrix(rotation_matrix)
# euler_angles_deg = rotations.as_euler('xyz', degrees=True)
# # print("歐拉角（度數）：", euler_angles_deg)
# %%

def EulerP2Rot(EulerP, Dir='G2L'):
    """
    將歐拉參數轉換為旋轉矩陣。
    
    Parameters
    ----------
    P : numpu.array
        包含歐拉參數的數組，每行包含 e0、e1、e2 和 e3。[n frames X 4].
    Dir : str
        轉換方向，'L2G' 表示局部到全局，'G2L' 表示全局到局部，The default is 'G2L'.
    
    
    Returns
    -------
    Rot : 
        旋轉矩陣.[3 X 3 X n frames]
    """
    e0 = EulerP[:, 0]
    e1 = EulerP[:, 1]
    e2 = EulerP[:, 2]
    e3 = EulerP[:, 3]

    Rot = np.zeros((3, 3, EulerP.shape[0]))

    for i in range(EulerP.shape[0]):
        r11 = e0[i]**2 + e1[i]**2 - 0.5
        r12 = e1[i] * e2[i] - e0[i] * e3[i]
        r13 = e1[i] * e3[i] + e0[i] * e2[i]
        r21 = e1[i] * e2[i] + e0[i] * e3[i]
        r22 = e0[i]**2 + e2[i]**2 - 0.5
        r23 = e2[i] * e3[i] - e0[i] * e1[i]
        r31 = e1[i] * e3[i] - e0[i] * e2[i]
        r32 = e2[i] * e3[i] + e0[i] * e1[i]
        r33 = e0[i]**2 + e3[i]**2 - 0.5
        if Dir == 'L2G':
            Rot[:, :, i] = 2 * np.array([[r11, r12, r13],
                                         [r21, r22, r23],
                                         [r31, r32, r33]])
        elif Dir == 'G2L':
            Rot[:, :, i] = 2 * np.array([[r11, r21, r31],
                                         [r12, r22, r32],
                                         [r13, r23, r33]])
    
    return Rot
# %%
def numerical_derivative(f, x, epsilon=1e-6):
    # 數值微分函數，計算 f 在 x 處的導數
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

def compute_rotation_matrix_derivative(R_func, t):
    # 計算旋轉矩陣在時間 t 的導數
    def derivative_at_t(x):
        return R_func(x)[t]
    
    R_derivative = np.array([
        [numerical_derivative(derivative_at_t, 0), numerical_derivative(derivative_at_t, 1), numerical_derivative(derivative_at_t, 2)],
        [numerical_derivative(derivative_at_t, 3), numerical_derivative(derivative_at_t, 4), numerical_derivative(derivative_at_t, 5)],
        [numerical_derivative(derivative_at_t, 6), numerical_derivative(derivative_at_t, 7), numerical_derivative(derivative_at_t, 8)]
    ])
    
    return R_derivative
# %% 從毆拉參數計算角速度及角加速度


def EulerP2Angular(EulerP, smprate, CS='local', method="continuous"):
    """
    計算局部或全局坐標系統的角速度和角加速度。
    
    Parameters
    ----------
        P：歐拉參數數組，每行包含 e0、e1、e2 和 e3。[n frames X 4]
        smprate：採樣率。
        CS：坐標系統，可以是 'local' 或 'global' 字符串. The default is 'local'
    
    
    Returns
    -------
        AngVel：角速度。
        AngAcc：角加速度。
    """
    # EulerP = P
    # EulerP = ElbowEulerP
    # smprate=180
    # 獲取數據的行數
    DataL = EulerP.shape[0]
    # 根據採樣率計算時間數組
    Time = np.arange(0, DataL / smprate, 1 / smprate)

    # 提取歐拉參數的各個分量
    e0 = EulerP[:, 0]
    e1 = EulerP[:, 1]
    e2 = EulerP[:, 2]
    e3 = EulerP[:, 3]
    if method == "continuous":
    # 初始化用於儲存內插數組的列表
        # Coef_P = []
        # Diff_Coef_P = []
        # Diff2_Coef_P = []
        P_dot = np.zeros((DataL, 4))
        P_dot2 = np.zeros((DataL, 4))
        # P_dot = np.zeros_like(EulerP)
        # P_dot2 = np.zeros_like(EulerP)
    
        # 使用三次樣條內插來生成歐拉參數的一階和二階導數
        for i in range(4):
            # cSpline = CubicSpline(Time, EulerP[:, i])
            spline  = UnivariateSpline(Time, EulerP[:, i], k=3, s=0.001)
            # 計算一階導數（速度）
            spline_derivative = spline.derivative(n=1)
            P_dot[:, i] = spline_derivative(Time)
            # cSpline_derivative = cSpline.derivative(nu=1)
            # P_dot[:, i] = cSpline_derivative(Time)
            # 計算二階導數（加速度）
            spline_derivative2 = spline.derivative(n=2)
            P_dot2[:, i] = spline_derivative2(Time)
            # Diff2_Coef_P.append(Coef_P[i].derivative(nu=2))
            # P_dot2[:, i] = Diff2_Coef_P[i](Time)

        # 初始化用於儲存局部和全局坐標系統的角速度和角加速度的數組
        Omega_L = np.zeros((DataL, 3))
        Alpha_L = np.zeros((DataL, 3))
        Omega_G = np.zeros((DataL, 3))
        Alpha_G = np.zeros((DataL, 3))
    
        # 計算角速度和角加速度
        for i in range(DataL):
            e_vec = np.array([e1[i], e2[i], e3[i]])
            e_wave = np.array([[0, -e3[i], e2[i]],
                               [e3[i], 0, -e1[i]],
                               [-e2[i], e1[i], 0]])
            E = np.hstack(((-e_vec[:, np.newaxis]),(e_wave + e0[i] * np.eye(3))))
            G = np.hstack(((-e_vec[:, np.newaxis]),(-e_wave + e0[i] * np.eye(3))))
            # E = np.concatenate(([-e_vec], e_wave + e0[i] * np.eye(3)), axis=1)
            # G = np.concatenate(([-e_vec], -e_wave + e0[i] * np.eye(3)), axis=1)
    
            Omega_L[i, :] = 2 * np.dot(G, P_dot[i, :].T)
            Alpha_L[i, :] = 2 * np.dot(G, P_dot2[i, :].T)
    
            Omega_G[i, :] = 2 * np.dot(E, P_dot[i, :].T)
            Alpha_G[i, :] = 2 * np.dot(E, P_dot2[i, :].T)
    
        # 根據指定的坐標系統返回角速度和角加速度
        if CS == 'local':
            AngVel = Omega_L
            AngAcc = Alpha_L
        elif CS == 'global':
            AngVel = Omega_G
            AngAcc = Alpha_G
    # 如果是使用離散方法處理數據
    elif method == "discrete":
        EulerP2Rot

    return AngVel, AngAcc
# %% 

def Rot2LocalAngularEP(Rot, smprate, place = "joint", unit="degree"):
    """
    

    Parameters
    ----------
    Rot : TYPE
        DESCRIPTION.
    smprate : TYPE
        DESCRIPTION.
    place : TYPE, optional
        DESCRIPTION. The default is "joint".
    unit : str, optional
        choosing the unit of output value is degree or rad

    Returns
    -------
    AngVel : TYPE
        DESCRIPTION.
    AngAcc : TYPE
        DESCRIPTION.

    """
    if place == "joint":
        new_Rot = np.empty(shape=np.shape(Rot))    
        for i in range(np.shape(Rot)[2]):
            new_Rot[:, :, i] = np.transpose(Rot[:, :, i])
        P, phi, n = Rot2EulerP(new_Rot)
    elif place == "segment":
        P, phi, n = Rot2EulerP(Rot)
    AngVel, AngAcc = EulerP2Angular(P, smprate)
    
    if unit == "degree":
        AngVel = np.degrees(AngVel)
        AngAcc = np.degrees(AngAcc)
    elif unit == "rad":
        AngVel = AngVel
    else:
        print("error : the unit is wrong")
    return AngVel, AngAcc

# %%
from scipy.integrate import solve_ivp

def euler_ode(t, y, omega):
    e0, e1, e2, e3 = y
    e_dot = -0.5 * np.array([
        [0, -omega[0], -omega[1], -omega[2]],
        [omega[0], 0, omega[2], -omega[1]],
        [omega[1], -omega[2], 0, omega[0]],
        [omega[2], omega[1], -omega[0], 0]
    ]) @ y
    # @ 矩陣乘法
    return e_dot

# Initial conditions
e0_init = 1.0
e1_init = 0.0
e2_init = 0.0
e3_init = 0.0
initial_state = [e0_init, e1_init, e2_init, e3_init]

# Angular velocity (in body frame)
omega = np.array([0.1, 0.2, 0.3])

# Time points for integration
t_start = 0
t_end = 10
t_points = np.linspace(t_start, t_end, 100)

# Solve ODE to get Euler parameters over time
solution = solve_ivp(
    fun=lambda t, y: euler_ode(t, y, omega),
    t_span=(t_start, t_end),
    y0=initial_state,
    t_eval=t_points,
    method='RK45'
)

# Calculate angular velocity and acceleration
e0_vals, e1_vals, e2_vals, e3_vals = solution.y
angular_velocity = 2 * np.array([
    e0_vals * e1_vals,
    e0_vals * e2_vals,
    e0_vals * e3_vals
]).T
angular_acceleration = 2 * np.array([
    e0_vals * (-omega[0] * e1_vals - omega[1] * e2_vals - omega[2] * e3_vals),
    e0_vals * (omega[0] * e0_vals - omega[2] * e2_vals + omega[1] * e3_vals),
    e0_vals * (omega[1] * e0_vals + omega[2] * e1_vals - omega[0] * e3_vals)
]).T

# angular_velocity and angular_acceleration are now available for each time point




# %%
def included_angle(x0, x1, x2):
    """
    計算由三個點所形成的夾角，以角度表示。

    Parameters
    ----------
    x0 : array-like
        第一個點的坐標。
    x1 : array-like
        第二個點的坐標（頂點）。
    x2 : array-like
        第三個點的坐標。

    Returns
    -------
    angle_degrees_360 : ndarray
        夾角的角度值，範圍在[0, 360]度之間。

    """
                              
    # 將輸入的點轉換為NumPy數組
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    # 計算向量A（從點x1到點x0的向量）
    vector_A = x0 - x1
    # 計算向量B（從點x1到點x2的向量）
    vector_B = x2 - x1
    
    # 計算向量A和向量B的點積
    dot_product = np.sum(vector_A * vector_B, axis=1)
    # 計算向量A的模長（即向量A的大小）
    magnitude_A = np.linalg.norm(vector_A, axis=1)
    # 計算向量B的模長（即向量B的大小）
    magnitude_B = np.linalg.norm(vector_B, axis=1)
    
    # 計算向量A和向量B的夾角的余弦值
    cosines = dot_product / (magnitude_A * magnitude_B)
    # 將余弦值裁剪到[-1, 1]之間，以避免反餘弦函數中出現無效值
    cosines = np.clip(cosines, -1, 1)
    
    # 計算夾角的弧度值
    angle_radians = np.arccos(cosines)
    # 將弧度值轉換為角度值
    angle_degrees = np.degrees(angle_radians)
    # 將角度值轉換到[0, 360]度範圍內
    angle_degrees_360 = (angle_degrees + 360) % 360
    
    # 返回最終的角度值
    return angle_degrees_360



