# %%
import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\function")
import gen_function as func

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
    骨盆坐標系定義: 旋轉順序 - XYZ
    O (Hip Joint Center): Tylkowski-Andriacchi method
        
        W = |R_ASIS - L_ASIS|
        
        R_Hip = (R_ASIS_x - 0.14*W,
                 R_ASIS_y - 0.19*W,
                 R_ASIS_z - 0.30*W)
        
        L_Hip = (L_ASIS_x + 0.14*W,
                 L_ASIS_y - 0.19*W,
                 L_ASIS_z - 0.30*W)
        
    z-axis: (R_ASIS - L_PSIS) X (L_ASIS - R_PSIS)/ |(R_ASIS - L_PSIS) X (L_ASIS - R_PSIS)|
    x-axis: (((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2) X y-axis) / 
            |(((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2) X y-axis)|
    y-axis: z-axis cross x-axis
    
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
    z_vector = (np.cross((R_ASIS - L_PSIS), (L_ASIS - R_PSIS)))/ \
        (np.linalg.norm(np.cross((R_ASIS - L_PSIS), (L_ASIS - R_PSIS))))
    x_vector = np.cross(((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2), z_vector)/ \
        (np.linalg.norm(np.cross(((R_ASIS + L_ASIS)/2 - (R_PSIS + L_PSIS)/2), z_vector)))
    y_vector = np.cross(z_vector, x_vector)/ \
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


# %% 定義大腿座標系
def DefCoordThigh(hip, Knee_Med, Knee_Lat, side = 'r'):
    """

    大腿骨坐標系定義:  旋轉順序 - XYZ
    
    O: Hip Joint Center
    y-axis : (O - 0.5*(Knee_Med + Knee_Lat))/
            |(O - 0.5*(Knee_Med + Knee_Lat))|
    v vector: (Knee_Lat - Knee_Med)/
            |(Knee_Lat - Knee_Med)|
    x-axis : y-axis cross v vector
            
    z-axis : x-axis cross y-axis
        
    R = [[ix, iy, iz],
         [jx, jy, jz],
         [kx, ky, kz]]

    Parameters
    ----------

    Returns
    -------
    RotMatrix : TYPE
        DESCRIPTION.


    """
    # 將 input 轉成 np.array
    hip = np.array(hip)
    Knee_Med = np.array(Knee_Med)
    Knee_Lat = np.array(Knee_Lat)
    
    # W = np.linalg.norm(R_ASIS - L_ASIS)
    # 定義座標軸方向
    y_vector = (hip - 0.5*(Knee_Med + Knee_Lat))/ \
        (np.linalg.norm((hip - 0.5*(Knee_Med + Knee_Lat))))
    v_vector = (Knee_Lat - Knee_Med)/ \
        np.linalg.norm((Knee_Lat - Knee_Med))

    x_vector = np.cross(y_vector, v_vector)/ \
        (np.linalg.norm(np.cross(y_vector, v_vector)))
        
    z_vector = (np.cross(x_vector, y_vector)) / \
          (np.linalg.norm(np.cross(x_vector, y_vector)))

    RotMatrix = np.array([x_vector, y_vector, z_vector])
        
    return RotMatrix

# %% 定義小腿坐標系

def DefCoordShank(Knee_Med, Knee_Lat, Ankle_Med, Ankle_Lat,
                  side = 'r'):
    """

    小腿坐標系定義:  旋轉順序 - XYZ
    
    O(Knee Joint Center): 0.5*(Knee_Med + Knee_Lat)
    y-axis : (O - 0.5*(Ankle_Med + Ankle_Lat))/
            |(O - 0.5*(Ankle_Med + Ankle_Lat))|
    v vector: (Knee_Lat - Knee_Med)/
            |(Knee_Lat - Knee_Med)|
    x-axis : y-axis cross v vector
            
    z-axis : x-axis cross y-axis
        
    R = [[ix, iy, iz],
         [jx, jy, jz],
         [kx, ky, kz]]

    Parameters
    ----------

    Returns
    -------
    RotMatrix : TYPE
        DESCRIPTION.
        

    """
    # 將 input 轉成 np.array
    Knee_Med = np.array(Knee_Med)
    Knee_Lat = np.array(Knee_Lat)
    Ankle_Med = np.array(Ankle_Med)
    Ankle_Lat = np.array(Ankle_Lat)
    
    o = 0.5*(Knee_Med + Knee_Lat)
    # 定義座標軸方向
    y_vector = (o - 0.5*(Ankle_Med + Ankle_Lat))/ \
        (np.linalg.norm((o - 0.5*(Ankle_Med + Ankle_Lat))))
    v_vector = (Knee_Lat - Knee_Med)/ \
        np.linalg.norm((Knee_Lat - Knee_Med))

    x_vector = np.cross(y_vector, v_vector)/ \
        (np.linalg.norm(np.cross(y_vector, v_vector)))
        
    z_vector = (np.cross(x_vector, y_vector)) / \
          (np.linalg.norm(np.cross(x_vector, y_vector)))

    RotMatrix = np.array([x_vector, y_vector, z_vector])
        
    return RotMatrix

# %% 定義小腿坐標系

def DefCoordFoot(Heel, Ankle_Med, Ankle_Lat,
                 side = 'r'):
    """

    小腿坐標系定義:  旋轉順序 - XYZ
    
    O(Heel): Heel
    y-axis : (O - 0.5*(Ankle_Med + Ankle_Lat))/
            |(O - 0.5*(Ankle_Med + Ankle_Lat))|
    v vector: (Knee_Lat - Knee_Med)/
            |(Knee_Lat - Knee_Med)|
    x-axis : y-axis cross v vector
            
    z-axis : x-axis cross y-axis
        
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
    Knee_Med = np.array(Knee_Med)
    Knee_Lat = np.array(Knee_Lat)
    Ankle_Med = np.array(Ankle_Med)
    Ankle_Lat = np.array(Ankle_Lat)
    
    o = 0.5*(Knee_Med + Knee_Lat)
    # 定義座標軸方向
    y_vector = (o - 0.5*(Ankle_Med + Ankle_Lat))/ \
        (np.linalg.norm((o - 0.5*(Ankle_Med + Ankle_Lat))))
    v_vector = (Knee_Lat - Knee_Med)/ \
        np.linalg.norm((Knee_Lat - Knee_Med))

    x_vector = np.cross(y_vector, v_vector)/ \
        (np.linalg.norm(np.cross(y_vector, v_vector)))
        
    z_vector = (np.cross(x_vector, y_vector)) / \
          (np.linalg.norm(np.cross(x_vector, y_vector)))

    RotMatrix = np.array([x_vector, y_vector, z_vector])
        
    return RotMatrix











