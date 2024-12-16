"""
Created on Mon Aug  7 18:16:31 2023

所含function

1. 關節座標定義
DefCoordArm: 定義上臂座標系
DefCoordForearm: 定義小臂座標系
DefCoordHand: 定義手掌座標系

joint_angle_rot: 計算兩坐標系之間的旋轉矩陣
transform_up_Z2Y: transform Z axis up to Y axis up
transformation_matrix: 將座標轉換成local or global

2. 旋轉矩陣轉換
Rot2EulerAngle: 將旋轉矩陣轉為尤拉角
Rot2EulerP: 將旋轉矩陣轉成毆拉參數
EulerP2Rot:將歐拉參數轉換為旋轉矩陣。
    
3. 角度計算
included_angle: 計算由三個點所形成的夾角，以角度表示。
Rot2LocalAngularEP: 旋轉矩陣 Rot 計算出角速度 (AngVel) 和角加速度 (AngAcc)
EulerP2Angular: 計算局部或全局坐標系統的角速度和角加速度。

@author: Hsin.YH.Yang
"""

# import os
import numpy as np
import pandas as pd
import sys
# 路徑改成你放自己code的資料夾
sys.path.append(r"C:\Users\Public\BenQ\myPyCode\U3")
import gen_function as gen
# from scipy.signal import find_peaks
# import gc
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import math
from scipy import signal
import gc

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
def DefCoordForearm(R_Elbow_Lat, R_Elbow_Med, R_Wrist_Una, R_Wrist_Rad, side='R'):
    """
    針對右手
    x-axis : R_Wrist_Rad - R_Wrist_Una
    y-axis : (R_Elbow_Lat + R_Elbow_Med)/2 - (R_Wrist_Una + R_Wrist_Rad)/2
    z-axis : x cross y
    
    左手定義需再更改
    
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
   
    if side == 'R':
 
        # 定義座標軸方向
        # j_vector = (R_Elbow_Lat + R_Elbow_Med)/2 - (R_Wrist_Una + R_Wrist_Rad)/2
        y_vector = (R_Elbow_Lat + R_Elbow_Med)/2 - R_Wrist_Una 
        x_vector = np.cross(y_vector, (R_Wrist_Rad - R_Wrist_Una))
        z_vector = np.cross(x_vector, y_vector)
        
        y_axis = y_vector / np.linalg.norm(y_vector)
        x_axis = x_vector / np.linalg.norm(x_vector)
        z_axis = z_vector / np.linalg.norm(z_vector)
        
        RotMatrix = np.array([x_axis, y_axis, z_axis])
    elif side == 'L':
        y_vector = R_Wrist_Una - (R_Elbow_Lat + R_Elbow_Med)/2
        x_vector = np.cross((R_Wrist_Rad - R_Wrist_Una), y_vector)
        z_vector = np.cross(x_vector, y_vector)
        
        y_axis = y_vector / np.linalg.norm(y_vector)
        x_axis = x_vector / np.linalg.norm(x_vector)
        z_axis = z_vector / np.linalg.norm(z_vector)
        
        RotMatrix = np.array([x_axis, y_axis, z_axis])
        
    return RotMatrix

## 定義手掌座標系
def DefCoordHand(R_Wrist_Una, R_Wrist_Rad, R_M_Finger1):
    """
    x-axis : y cross (R_Wrist_Rad - R_Wrist_Una)
    y-axis : (R_Wrist_Una + R_Wrist_Rad)/2 - R_M_Finger1
    z-axis : x cross y
    
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

# %% Transformation between Coordinate System
"""
1. define Local Coordinate System.
    1.1. define LCS orgin. 
    1.2. calculate Local Coordinate.
Transformation matrix :
    1. Linear transformation.
    2. Rotational Transformation.
"""


def transformation_matrix(LCS_0, LCS_1, LCS_2, p, O, rotation="GCStoLCS"):
    """
    Parameters
    ----------
    LCS_0 : np.array
        The orgin of LCS.
    LCS_1 : np.array
        To create long axis with respect to orgin point.
    LCS_2 : np.array
        To create plane axis with respect to orgin point.
    p : np.array
        the specific point coordinate with respect to GCS/LCS.
    rotation : Str, optional
        To determinate to rotation sequence of transform matrix. The default is 'GCStoLCS'.

    Returns
    -------
    p1 : np.array
        the specific point coordinate with respect to LCS/GCS.

    """
    
    # determinate the axis
    v1 = LCS_1 - LCS_0  # long axis
    v2 = np.cross(v1, (LCS_2 - LCS_0))  # superior direction
    v3 = np.cross(v1, v2)  # lateral direction

    # normalize the vector
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # calculate rotation matrix
    rotation_LCS = np.array([v1, v2, v3])  # or np.vstack((v1, v2, v3)).T
    # print("\nRotation matrix rotation_LCS:\n", rotation_LCS)

    if rotation == "GCStoLCS":
        p1 = np.matmul(rotation_LCS, (p - LCS_0))
    elif rotation == "LCStoGCS":
        p1 = np.matmul((np.transpose(rotation_LCS)), p) + LCS_0

    return p1

# %% transform Z axis up to Y axis up


def transform_up_Z2Y(raw_data, dtype="DataFrame"):
    """
    transform the axis of motion data from Z-up to Y-up.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Input raw motion data.
    dtype : Str, optional
    The default is 'DataFrame'.

    Returns
    -------
    data : pandas.DataFrame
        output the motion data has been transformed.

    """
    if dtype == "DataFrame":
        data = pd.DataFrame(np.zeros([np.shape(raw_data)[0], np.shape(raw_data)[1]]))
        for i in range(int((np.shape(raw_data.iloc[:, :])[1] - 2) / 3)):
            data.iloc[:, 2 + 3 * i] = raw_data.iloc[:, 2 + 3 * i]  # x -> x
            data.iloc[:, 3 + 3 * i] = raw_data.iloc[:, 4 + 3 * i]  # y -> z
            data.iloc[:, 4 + 3 * i] = -raw_data.iloc[:, 3 + 3 * i]  # z -> -y
    elif dtype == "np.array":
        print("rotation np.array")
        # raw_data = new_trun_motion
        data = np.empty(shape=(np.shape(raw_data)), dtype=float)
        for i in range(int((np.shape(raw_data)[0]))):
            data[i, :, 0] = raw_data[i, :, 0]  # x -> x
            data[i, :, 1] = raw_data[i, :, 2]  # y -> z
            data[i, :, 2] = -raw_data[i, :, 1]  # z -> -y
    else:
        print("rotation fail")
    return data
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
# %%  旋轉矩陣 Rot 計算出角速度 (AngVel) 和角加速度 (AngAcc)

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
def included_angle(x0, x1, x2, x3=None):
    """
    計算由三個點所形成的夾角，以角度表示。

    Parameters
    ----------
    x0 : array-like
        第一個點的坐標。(四個點：向量1起始)
    x1 : array-like
        第二個點的坐標（三個點：頂點）。
    x2 : array-like
        第三個點的坐標。(四個點：向量2起始)
    if x3 == true
    x3 : array-like
        第四個點的坐標。

    Returns
    -------
    angle_degrees_360 : ndarray
        夾角的角度值，範圍在[0, 360]度之間。

    """
                              
    # 將輸入的點轉換為NumPy數組
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    if x3 is None:
        # 三個點的情況：以中間點為頂點計算夾角
        vector_A = x0 - x1  # 向量A從p2指向p1
        vector_B = x2 - x1  # 向量B從p2指向p3
    else:
        # 四個點的情況：計算兩個向量的夾角
        x3 = np.array(x3)
        vector_A = x1 - x0  # 向量A從p1指向p2
        vector_B = x3 - x2  # 向量B從p3指向p4
    
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


# %% 計算內上髁在LCS的位置
def V_Elbow_cal(c3d_path, method=None, replace=None):
    # c3d_path = r'E:\Hsin\BenQ\ZOWIE non-sym\\1.motion\Vicon\S03\S03_Tpose_hand.c3d'
    motion_info, motion_data, analog_info, FP_data, np_motion_data = gen.read_c3d(c3d_path, method=method)
    if replace:
        motion_data.rename(columns=lambda x: x.replace(str(replace + ':'), ''), inplace=True)
    # 1. 設定輸入計算 Virtual marker 參數 : 手肘內上髁, 外上髁, UA1, UA3
    R_Elbow_Med = motion_data.loc[:, "R.Elbow.Med_x":"R.Elbow.Med_z"].dropna(axis=0)
    R_Elbow_Lat = motion_data.loc[:, "R.Elbow.Lat_x":"R.Elbow.Lat_z"].dropna(axis=0)
    UA1 = motion_data.loc[:, "UA1_x":"UA1_z"].dropna(axis=0)
    UA3 = motion_data.loc[:, "UA3_x":"UA3_z"].dropna(axis=0)
    # 2. 避免數量中出現NAN，請造成不同變數見長短不一致，因此找出最短數列的 index
    ind_frame = min([np.shape(R_Elbow_Med)[0], np.shape(R_Elbow_Lat)[0], np.shape(UA1)[0], np.shape(UA3)[0]])
    for i in [R_Elbow_Med, R_Elbow_Lat, UA1, UA3]:
        if ind_frame == np.shape(i)[0]:
            ind_frame = i.index
            break
    # 3. 計算手肘內上髁在 LCS 之位置
    p1_all = pd.DataFrame(np.zeros([len(ind_frame), 3]))
    for frame in ind_frame:
        p1_all.iloc[frame :] = (transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
                                                    R_Elbow_Med.iloc[frame, :].values, np.array([0, 0, 0]),
                                                    rotation='GCStoLCS'))
            # 4. 清除不需要的變數
    del motion_info, motion_data, analog_info, FP_data, R_Elbow_Med, UA1, UA3, np_motion_data
    gc.collect()
    return p1_all
# %% 計算橈側內髁之位置
def V_Elbow_cal_1(c3d_path, method=None, replace=None):
    motion_info, motion_data, analog_info, FP_data, np_motion_data = gen.read_c3d(c3d_path, method=method)
    if replace:
        motion_data.rename(columns=lambda x: x.replace(str(replace + ':'), ''), inplace=True)
    # 1. 設定輸入計算 Virtual marker 參數 : 手肘內上髁, 外上髁, UA1, UA3
    R_Elbow_Med = motion_data.loc[:, "R.Elbow.Med_x":"R.Elbow.Med_z"].dropna(axis=0)
    R_Elbow_Lat = motion_data.loc[:, "R.Elbow.Lat_x":"R.Elbow.Lat_z"].dropna(axis=0)
    UA1 = motion_data.loc[:, "UA1_x":"UA1_z"].dropna(axis=0)
    UA3 = motion_data.loc[:, "UA3_x":"UA3_z"].dropna(axis=0)
    # 2. 避免數量中出現NAN，請造成不同變數見長短不一致，因此找出最短數列的 index
    ind_frame = min([np.shape(R_Elbow_Med)[0], np.shape(R_Elbow_Lat)[0], np.shape(UA1)[0], np.shape(UA3)[0]])
    for i in [R_Elbow_Med, R_Elbow_Lat, UA1, UA3]:
        if ind_frame == np.shape(i)[0]:
            ind_frame = i.index
            break
    # 3. 計算手肘內上髁在 LCS 之位置
    p1_all = pd.DataFrame(np.zeros([len(ind_frame), 3]))
    for frame in ind_frame:
        p1_all.iloc[frame :] = (transformation_matrix(R_Elbow_Lat.iloc[frame, :].values, UA1.iloc[frame, :].values, UA3.iloc[frame, :].values,
                                                           R_Elbow_Med.iloc[frame, :].values, np.array([0, 0, 0]),
                                                           rotation='GCStoLCS'))
            # 4. 清除不需要的變數
    del motion_info, motion_data, analog_info, FP_data, R_Elbow_Med, UA1, UA3, np_motion_data
    gc.collect()
    return p1_all

# %% 使用tpose計算手部的自然關節角度
def arm_natural_pos(c3d_path, p1_all, index, method=None, replace=None):
    # c3d_path = r'D:/BenQ_Project/01_UR_lab/2024_11 Shanghai CS Major/1. Motion/Major_Asymmetric/S01/20241123/S01_Tpose_hand.c3d'
    motion_info, motion_data, analog_info, FP_data, np_motion_data = gen.read_c3d(c3d_path, method=method)
    if replace:
        motion_data.rename(columns=lambda x: x.replace(str(replace + ':'), ''), inplace=True)
    V_R_Elbow_Med = np.zeros(shape=(3))
    # 計算虛擬手肘內上髁位置
    V_R_Elbow_Med[:] = transformation_matrix(motion_data.loc[index, "R.Elbow.Lat_x":"R.Elbow.Lat_z"].values, # R.Elbow.Lat
                                             motion_data.loc[index, "UA1_x":"UA1_z"].values, # UA1
                                             motion_data.loc[index, "UA3_x":"UA3_z"].values, # UA3
                                             p1_all.iloc[5, :].values, np.array([0, 0, 0]),
                                             rotation='LCStoGCS')
    # 定義手部支段坐標系
    static_ArmCoord = np.empty(shape=(3, 3))
    static_ForearmCoord = np.empty(shape=(3, 3))
    static_HandCoord = np.empty(shape=(3, 3))
    # 定義人體自然角度坐標系, tpose 手部自然放置角度
    static_ArmCoord[:, :] = DefCoordArm(motion_data.loc[index, "R.Shoulder_x":"R.Shoulder_z"],
                                        motion_data.loc[index, "R.Elbow.Lat_x":"R.Elbow.Lat_z"],
                                        V_R_Elbow_Med[:])
    static_ForearmCoord[:, :] = DefCoordForearm(motion_data.loc[index, "R.Elbow.Lat_x":"R.Elbow.Lat_z"],
                                                V_R_Elbow_Med[:],
                                                motion_data.loc[index, "R.Wrist.Uln_x":"R.Wrist.Uln_z"],
                                                motion_data.loc[index, "R.Wrist.Rad_x":"R.Wrist.Rad_z"])
    static_HandCoord[:, :] = DefCoordHand(motion_data.loc[index, "R.Wrist.Uln_x":"R.Wrist.Uln_z"],
                                          motion_data.loc[index, "R.Wrist.Rad_x":"R.Wrist.Rad_z"],
                                          motion_data.loc[index, "R.M.Finger1_x":"R.M.Finger1_z"])
    # 清除不需要的變數
    del motion_info, motion_data, analog_info, FP_data, np_motion_data, index
    gc.collect()
    return static_ArmCoord, static_ForearmCoord, static_HandCoord
# %% 計算大臂, 小臂, 手掌隨時間變化的坐標系

def UpperExtremty_coord(trun_motion, motion_info, p1_all):
    
    # trun_motion = trun_motion_np#, motion_info, p1_all
    # 1.2.5. ---------計算手肘內上髁之位置----------------------------------
    # 建立手肘內上髁的資料貯存位置
    V_R_Elbow_Med = np.zeros(shape=(1, np.shape(trun_motion)[1], np.shape(trun_motion)[2]))
    # 找出以下三個字串的索引值
    target_strings = ["R.Elbow.Lat", "UA1"
                      , "UA3", "R.Shoulder",
                      "R.Wrist.Uln", "R.Wrist.Rad",
                      "R.M.Finger1"]
    indices = []
    for target_str in target_strings:
        try:
            index = motion_info["LABELS"].index(target_str)
            indices.append(index)
        except ValueError:
            indices.append(None)
    # 回算手肘內上髁在 GCS 之位置
    for frame in range(np.shape(trun_motion)[1]):
        V_R_Elbow_Med[0, frame, :] = transformation_matrix(trun_motion[indices[0], frame, :], # R.Elbow.Lat
                                                           trun_motion[indices[1], frame, :], # UA1
                                                           trun_motion[indices[2], frame, :], # UA3
                                                           p1_all.iloc[5, :].values, np.array([0, 0, 0]),
                                                           rotation='LCStoGCS')
    # 合併 motion data and virtual R.Elbow.Med data
    new_trun_motion = np.concatenate((trun_motion, V_R_Elbow_Med), axis=0)
    # motion_info 新增 R.Elbow.Med 的標籤
    motion_info['LABELS'].append("R.Elbow.Med")
    # 去掉LABELS中關於Cortex Marker set 之資訊 -> 去掉 EC2 Wight:
    for label in range(len(motion_info['LABELS'])):
        motion_info['LABELS'][label] = motion_info['LABELS'][label].replace("EC2 Wight:", "")
    # 去除掉 Virtual marker 的 LABELS 與 motion data
    new_labels = []
    np_labels = []
    for key, item in enumerate(motion_info['LABELS']):
        # print(key, item)
        # 要去除的特定字符 : V_marker
        if "V_" not in item:
            new_labels.append(item)
            np_labels.append(key) # add time and Frame
    motion_info['LABELS'] = new_labels
    # 重新定義 motion data
    new_np_motion_data = new_trun_motion[np_labels, :, :]
    # # 低通濾波 butterworth filter
    bandpass_filtered = np.empty(shape=np.shape(new_np_motion_data))
    bandpass_sos = signal.butter(2, 6/0.802,  btype='lowpass', fs=motion_info["frame_rate"], output='sos')
    for iii in range(np.shape(new_np_motion_data)[0]):
        for iiii in range(np.shape(new_np_motion_data)[2]):
            bandpass_filtered[iii, :, iiii] = signal.sosfiltfilt(bandpass_sos,
                                                                 new_np_motion_data[iii, :, iiii])
    
    # 開始計算運動學資料
    ArmCoord = np.empty(shape=(3, 3, np.shape(bandpass_filtered)[1]))
    ForearmCoord = np.empty(shape=(3, 3, np.shape(bandpass_filtered)[1]))
    HandCoord = np.empty(shape=(3, 3, np.shape(bandpass_filtered)[1]))
    # 對每個 Frame 定義坐標系
    for i in range(np.shape(bandpass_filtered)[1]):
        ArmCoord[:, :, i] = DefCoordArm(bandpass_filtered[indices[3], i, :], # R.Shoulder
                                            bandpass_filtered[indices[0], i, :], # R.Elbow.Lat
                                            bandpass_filtered[-1, i, :]) # R.Elbow.Med
        ForearmCoord[:, :, i] = DefCoordForearm(bandpass_filtered[indices[0], i, :], # R.Elbow.Lat
                                                bandpass_filtered[-1, i, :], # R.Elbow.Med
                                                bandpass_filtered[indices[4], i, :], # R.Wrist.Uln
                                                bandpass_filtered[indices[5], i, :]) # R.Wrist.Rad
        HandCoord[:, :, i] = DefCoordHand(bandpass_filtered[indices[4], i, :], # R.Wrist.Uln
                                            bandpass_filtered[indices[5], i, :], # R.Wrist.Rad
                                            bandpass_filtered[indices[6], i, :]) # R.M.Finger1
    return ArmCoord, ForearmCoord, HandCoord, motion_info, bandpass_filtered
# %% 計算手指關節角度
def finger_angle_cal(file_name, motion_data, motion_info):
    # 建立要尋找的motion data label
    target_strings = ["R.Wrist.Rad", "R.Wrist.Uln",
                      "R.Thumb1", "R.Thumb2",
                      "R.I.Finger1", "R.I.Finger2", "R.I.Finger3",
                      "R.M.Finger1", "R.M.Finger2",
                      "R.R.Finger1", "R.R.Finger2",
                      "R.P.Finger1", "R.P.Finger2"]
    # 找出指定motion data label的索引
    indices = []
    for target_str in target_strings:
        try:
            index = motion_info["LABELS"].index(target_str)
            indices.append(index)
        except ValueError:
            indices.append(None)
    # 建立暫存的矩陣
    hand_angle_table = pd.DataFrame({}, columns=["filename", "CMP1",
                                                 "CMP2", "PIP2",
                                                 "CMP3", "PIP3",
                                                 "CMP4", "CMP5"])
            


    wrist_cen = (motion_data[indices[0], :, :] + motion_data[indices[1], :, :])/2
    CMP1 = included_angle(wrist_cen,                         # wrist
                              motion_data[indices[2], :, :], # R.Thumb1
                              motion_data[indices[3], :, :]) # R.Thumb2
    CMP2 = included_angle(wrist_cen,                         # wrist
                              motion_data[indices[4], :, :], # R.I.Finger1
                              motion_data[indices[5], :, :]) # R.I.Finger2
    PIP2 = included_angle(motion_data[indices[4], :, :], # R.I.Finger1
                              motion_data[indices[5], :, :], # R.I.Finger2
                              motion_data[indices[6], :, :]) # R.I.Finger3
    CMP3 = included_angle(wrist_cen,                         # wrist
                              motion_data[indices[7], :, :], # R.M.Finger1
                              motion_data[indices[8], :, :]) # R.M.Finger2
    CMP4 = included_angle(wrist_cen,                         # wrist
                              motion_data[indices[9], :, :], # R.R.Finger1
                              motion_data[indices[10], :, :]) # R.R.Finger2
    CMP5 = included_angle(wrist_cen,                          # wrist
                              motion_data[indices[11], :, :], # R.P.Finger1
                              motion_data[indices[12], :, :]) # R.P.Finger2
    hand_angle_table = pd.concat([hand_angle_table,
                                  pd.DataFrame({"filename":file_name,
                                                "CMP1":np.mean(CMP1),
                                                "CMP2":np.mean(CMP2),
                                                "PIP2":np.mean(PIP2),
                                                "CMP3":np.mean(CMP3),
                                                "CMP4":np.mean(CMP4),
                                                "CMP5":np.mean(CMP5)
                                                },index=[0])],
                                 ignore_index=True)
    return hand_angle_table
