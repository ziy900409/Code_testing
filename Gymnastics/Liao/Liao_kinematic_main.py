# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:21:31 2025

@author: Hsin.YH.Yang
"""
import numpy as np
import pandas as pd
import sys
# 路徑改成你放自己code的資料夾
# sys.path.append(r"E:\Hsin\git\git\Code_testing\LabProject\function")
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\function")
import gen_function as func
import Kinematic_function as kincal
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


# %%

clo_convert = {'C7': 'C7',
               'CLAV': 'Clavicle',
               'LANK': 'L.Ankle.Lat',
               'LASI': 'L.ASIS',
               'LBAK': 'L.Scap.Med',
               'LBHD': 'L.Rear.Head',
               'LELB': 'L.Epi.Lat',
               'LFHD': 'L.Front.Head',
               'LFIN': 'L.Hand',
               'LFRM': 'L.Forearm',
               'LHEE': 'L.Heel',
               'LKNE': 'L.Knee.Lat',
               'LMANK': 'L.Ankle.Med',
               'LMELB': 'L.Epi.Med',
               'LMKNE': 'L.Knee.Med',
               'LPSI': 'L.PSIS',
               'LSHO': 'L.Shoulder',
               'LTHI': 'L.Thigh',
               'LTIB': 'L.Shank',
               'LTOE': 'L.Toe',
               'LUPA': 'L.Upperarm',
               'LWRA': 'L.Wrist.Rad',
               'LWRB': 'L.Wrist.Uln',
               'RANK': 'R.Ankle.Lat',
               'RASI': 'R.ASIS',
               'RBAK': 'R.Scap.Med',
               'RBHD': 'R.Rear.Head',
               'RELB': 'R.Epi.Lat',
               'RFHD': 'R.Front.Head',
               'RFIN': 'R.Hand',
               'RFRM': 'R.Forearm',
               'RHEE': 'R.Heel',
               'RKNE': 'R.Knee',
               'RMANK': 'R.Knee.Med',
               'RMELB': 'R.Epi.Med',
               'RMKNE': 'R.Ankle.Med',
               'RPSI': 'R.PSIS',
               'RSHO':'R.Shoulder',
               'RTHI': 'R.Thigh',
               'RTIB': 'R.Shank',
               'RTOE': 'R.Toe',
               'RUPA': 'R.Upperarm',
               'RWRA': 'R.Wrist.Rad',
               'RWRB': 'R.Wrist.Uln',
               'STRN': 'STRN',
               'T10': 'T10'}
# %%
motion_path = r"C:\Users\Hsin.YH.Yang\Downloads\000511_001688_78_235_003_FF_931.c3d"

motion_info, motion_data, analog_info, analog_data, np_motion_data = func.read_c3d(motion_path,
                                                                                   method='vicon')


motion_data.loc[:, 'R_ASIS_x':'R_ASIS_z']
## 定義骨盆坐標系
def DefCoordPelvis(R_ASIS, L_ASIS, R_PSIS, L_PSIS, side = 'r'):
    """
    骨盆坐標系定義
    x-axis : (((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2) X y-axis) / 
            |(((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2) X y-axis)|
    y-axis : z cross x
    z-axis : (R_ASIS - L_PSIS) X (L_ASIS - R_PSIS)/ |(R_ASIS - L_PSIS) X (L_ASIS - R_PSIS)|
    
    
    Hip Joint Center (O): Tylkowski-Andriacchi method
        
        W = |R_ASIS - L_ASIS|
        
        R_Hip = (R_ASIS_x - 0.14*W,
                 R_ASIS_y - 0.19*W,
                 R_ASIS_z - 0.30*W)
        
        L_Hip = (L_ASIS_x + 0.14*W,
                 L_ASIS_y - 0.19*W,
                 L_ASIS_z - 0.30*W)
    
    R = [[ix, iy, iz],
         [jx, jy, jz],
         [kx, ky, kz]]

    Parameters
    ----------

    Returns
    -------
    RotMatrix : TYPE
        DESCRIPTION.
        
    Reference:
        1. https://www.wiki.has-motion.com/doku.php?id=visual3d:documentation:kinematics_and_kinetics:joint

    """
    # 將 input 轉成 np.array
    R_ASIS = np.array(R_ASIS)
    L_ASIS = np.array(L_ASIS)
    R_PSIS = np.array(R_PSIS)
    L_PSIS = np.array(L_PSIS)
    
    W = np.linalg.norm(R_ASIS - L_ASIS)
    # 定義座標軸方向
    z_vector = (np.cross((R_ASIS - L_PSIS), (L_ASIS - R_PSIS))) \
        (np.linalg.norm(np.cross((R_ASIS - L_PSIS), (L_ASIS - R_PSIS))))
    x_vector = np.cross(((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2), z_vector) \
        (np.linalg.norm(np.cross(((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2), z_vector)))
    y_vector = np.cross(z_vector, x_vector) \
        (np.linalg.norm(np.cross(z_vector, x_vector)))
    RotMatrix = np.array([x_vector, y_vector, z_vector])
    
    if side == 'R':
        hip = (R_ASIS[:, :, 0] - 0.14*W,
               R_ASIS[:, :, 1] - 0.19*W,
               R_ASIS[:, :, 2] - 0.30*W)
    elif side == 'L':
        hip = (L_ASIS[:, :, 0] + 0.14*W,
               L_ASIS[:, :, 1] - 0.19*W,
               L_ASIS[:, :, 2] - 0.30*W)
        
    return RotMatrix, hip

# %%
"""
5個分期點：啟動瞬間S、下蹲結束瞬間D、起跳瞬間T、展體瞬間O、著地瞬間L
分期方法1_力板
    分期點（一）：啟動瞬間S，使用Average小於5*SD做為啟動瞬間
分期方法2_ Motion
    分期點（二）：下蹲結束瞬間D，使用Hip Angle最屈曲瞬間
    分期點（三）：起跳瞬間T， 使用Hip Angle最伸展瞬間
    分期點（四）：展體瞬間O，使用Knee Angle第二次(在空中)伸展瞬間
    分期點（五）：著地瞬間L，地面反作用力大於”某數值"的第一瞬間

"""

"""
1. 同步訊號

力版採樣訊號只有1000
motion 240 hz
"""


# read staging file

# starting frame(motion)
starting_frame = 16664/10
# read motion, force plate, anc file
# 找到互相對應的檔名
motion_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\論文資料CSV檔\論文資料CSV檔\MOTION\NSF11__1_ok_20250115.data.csv",
                          skiprows=2)
fp_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\論文資料CSV檔\論文資料CSV檔\FORCE PLATE\force_NSF11_BTS_2_ok.csv",
                      skiprows=4)
anc_data = pd.read_csv(r"C:\Users\Hsin.YH.Yang\Downloads\論文資料CSV檔\論文資料CSV檔\FORCE PLATE\anc_NSF11_BTS_2.csv",
                       skiprows=8)
# read EMG file

# 找 trigger 訊號
trigger_signal = anc_data.loc[2:, 'C63'].reset_index(drop=True)
peaks, _ = find_peaks(trigger_signal, height=np.mean(trigger_signal)*3)
plt.plot(trigger_signal)
plt.plot(peaks, trigger_signal[peaks], "x")
plt.plot(np.zeros_like(trigger_signal), "--", color="gray")
plt.show()
trigger = peaks[0]

# 定義下肢運動學
# R Hip Flex / Ext Joint Angle (deg)
r_hip_ext = motion_data.loc[1:, 'R Hip Flex / Ext Joint Angle (deg)'].reset_index(drop=True)
# R Knee Flex / Ext Joint Angle (deg)
r_knee_ext = motion_data.loc[1:, 'R Knee Flex / Ext Joint Angle (deg)'].reset_index(drop=True)

# 定義力版訊號
fp1_z = fp_data.loc[:, 'FZ1']
fp2_z = fp_data.loc[:, 'FZ2']

# 分期點（一）：啟動瞬間S，使用Average小於5*SD做為啟動瞬間
# 找兩力板的訊號
fp1_z = fp_data.loc[:, 'FZ1']

# 分期點（二）：下蹲結束瞬間D，使用Hip Angle最屈曲瞬間



# 分期點（三）：起跳瞬間T， 使用Hip Angle最伸展瞬間


# 分期點（四）：展體瞬間O，使用Knee Angle第二次(在空中)伸展瞬間
# R Knee Flex / Ext Joint Angle (deg)

# 分期點（五）：著地瞬間L，地面反作用力大於”某數值"的第一瞬間
















