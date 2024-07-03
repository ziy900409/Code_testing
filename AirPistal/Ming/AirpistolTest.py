# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:55:22 2022

@author: user
"""

# Import the necessary libraries
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from IPython import get_ipython
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

S2_data_path = r'F:\HsinYang\NTSU\TenLab\Shooting\S2_2.trc'

S2_data = pd.read_csv(S2_data_path, delimiter='\t' ,skiprows=3, encoding='UTF-8')

S5_data_path = r'F:\HsinYang\NTSU\TenLab\Shooting\S5_1.trc'

S5_data = pd.read_csv(S5_data_path, delimiter='\t' ,skiprows=3, encoding='UTF-8')

S2_shooting = 3739
S5_shooting = 2685


S2_time = S2_data.iloc[1:, 1].values
S2_time = S2_time

S5_time = S5_data.iloc[1:, 1].values
S2_disp_x = S2_data.iloc[S2_shooting-240:S2_shooting+120, 128]

# S2_disp_x = pd.DataFrame(S2_disp_x)
# S2_disp_x = pd.to_numeric(S2_disp_x.values)
S2_disp_y = S2_data.iloc[S2_shooting-240:S2_shooting+120, 129]
S2_disp_z = S2_data.iloc[S2_shooting-240:S2_shooting+120:, 130]

S5_disp_x = S5_data.iloc[S5_shooting-240:S5_shooting+120:, 128]
S5_disp_y = S5_data.iloc[S5_shooting-240:S5_shooting+120:, 129]
S5_disp_z = S5_data.iloc[S5_shooting-240:S5_shooting+120:, 130]

S5_disp_x_new = np.float_([x for x in S5_disp_x.astype('float64') if np.isnan(x) == False])
S5_disp_y_new = np.float_([x for x in S5_disp_y.astype('float64') if np.isnan(x) == False])
S5_disp_z_new = np.float_([x for x in S5_disp_z.astype('float64') if np.isnan(x) == False])

# plot data
# fig, (ax1,ax2) = plt.subplots(1, 2, sharex = True, figsize=(11, 4))
# plt.suptitle("Pezzack's benchmark data", fontsize=20)
# ax1.plot(S2_time, S2_disp_x, 'b')
# ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Angular displacement [rad]')pip
# ax2.plot(S2_time, S2_disp_y, 'g')
# ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Angular acceleration [rad/s$^2$]')
# plt.subplots_adjust(wspace=0.3)

from optcutfreq import optcutfreq

freq = np.mean(1/np.diff(S2_time.astype('float64')))

freq_1 = 1/(S2_time.astype('float64')[1] - S2_time.astype('float64')[0])
freq_2 = 1/(S5_time.astype('float64')[1] - S5_time.astype('float64')[0])

fc_opt = optcutfreq(S2_disp_x.astype('float64'), freq=freq_1, show=True)

fc_opt = optcutfreq(S2_disp_y.astype('float64'), freq=freq_1, show=True)

fc_opt = optcutfreq(S2_disp_z.astype('float64'), freq=freq_1, show=True)

fc_opt = optcutfreq(S5_disp_x_new, freq=freq_2, show=True)

fc_opt = optcutfreq(S5_disp_y_new, freq=freq_2, show=True)

fc_opt = optcutfreq(S5_disp_z_new, freq=freq_2, show=True)

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------


S16_data_path = r'F:\HsinYang\NTSU\TenLab\Shooting\S16_2.trc'

S16_data = pd.read_csv(S16_data_path, delimiter='\t' ,skiprows=3, encoding='UTF-8')

S24_data_path = r'F:\HsinYang\NTSU\TenLab\Shooting\S24_Shooting_2.trc'

S24_data = pd.read_csv(S24_data_path, delimiter='\t' ,skiprows=3, encoding='UTF-8')

S16_shooting = int(94 + 10.3*240)
S24_shooting = int(69 + 10.226*240)


S16_time = S16_data.iloc[1:, 1].values
S16_time = S16_time

S24_time = S24_data.iloc[1:, 1].values
S16_disp_x = S16_data.iloc[S16_shooting-240:S16_shooting+120, 128]

# S2_disp_x = pd.DataFrame(S2_disp_x)
# S2_disp_x = pd.to_numeric(S2_disp_x.values)
S16_disp_y = S16_data.iloc[S16_shooting-240:S16_shooting+120, 129]
S16_disp_z = S16_data.iloc[S16_shooting-240:S16_shooting+120:, 130]

S24_disp_x = S24_data.iloc[S24_shooting-240:S24_shooting+120:, 128]
S24_disp_y = S24_data.iloc[S24_shooting-240:S24_shooting+120:, 129]
S24_disp_z = S24_data.iloc[S24_shooting-240:S24_shooting+120:, 130]

S24_disp_x_new = np.float_([x for x in S24_disp_x.astype('float64') if np.isnan(x) == False])
S24_disp_y_new = np.float_([x for x in S24_disp_y.astype('float64') if np.isnan(x) == False])
S24_disp_z_new = np.float_([x for x in S24_disp_z.astype('float64') if np.isnan(x) == False])

# plot data
# fig, (ax1,ax2) = plt.subplots(1, 2, sharex = True, figsize=(11, 4))
# plt.suptitle("Pezzack's benchmark data", fontsize=20)
# ax1.plot(S2_time, S2_disp_x, 'b')
# ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Angular displacement [rad]')pip
# ax2.plot(S2_time, S2_disp_y, 'g')
# ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Angular acceleration [rad/s$^2$]')
# plt.subplots_adjust(wspace=0.3)


freq = np.mean(1/np.diff(S2_time.astype('float64')))

freq_1 = 1/(S16_time.astype('float64')[1] - S16_time.astype('float64')[0])
freq_2 = 1/(S24_time.astype('float64')[1] - S24_time.astype('float64')[0])

fc_opt = optcutfreq(S16_disp_x.astype('float64'), freq=freq_1, show=True)

fc_opt = optcutfreq(S16_disp_y.astype('float64'), freq=freq_1, show=True)

fc_opt = optcutfreq(S16_disp_z.astype('float64'), freq=freq_1, show=True)

fc_opt = optcutfreq(S24_disp_x_new, freq=freq_2, show=True)

fc_opt = optcutfreq(S24_disp_y_new, freq=freq_2, show=True)

fc_opt = optcutfreq(S24_disp_z_new, freq=freq_2, show=True)
