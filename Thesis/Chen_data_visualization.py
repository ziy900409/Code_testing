# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## -----------------data visualizlation
# read data
EMG_data_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing\iMVC\Shooting'
Release_time_file = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing\ReleaseTiming.xlsx'
Save_figuare = str(r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing')
Release_time_data = pd.read_excel(Release_time_file)
EMG_path_list = os.listdir(EMG_data_folder)
EMG_data = []
# TrapeziusUpper_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# Infraspinatis_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# LatissimusDorsi_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# Biceps_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# Triceps_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# Extensor_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# Flexor_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# TrapeziusLower_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# L_TrapeziusUpper_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# L_Biceps_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# L_Extensor_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# L_MedDeltoid_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# R_Deltoid_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])
# L_Deltoid_muscle = np.zeros([6000, np.shape(EMG_path_list)[0]])

# for i in range(np.shape(EMG_path_list)[0]):    
#     EMG_data = pd.read_excel(EMG_data_folder + '\\' + EMG_path_list[i])
#     Release_time = Release_time_data.iloc[i, 1]
#     print(Release_time)
#     TrapeziusUpper_muscle[:, i] = EMG_data.iloc[Release_time-5000:Release_time+1000,1]
#     Infraspinatis_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,2]
#     LatissimusDorsi_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,3]
#     Biceps_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,4]
#     Triceps_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,5]
#     Extensor_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,6]
#     Flexor_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,7]
#     TrapeziusLower_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,8]
#     L_TrapeziusUpper_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,9]
#     L_Biceps_muscle[:,i]= EMG_data.iloc[Release_time-5000:Release_time+1000,10]
#     L_Extensor_muscle[:,i]= EMG_data.iloc[Release_time-5000:Release_time+1000,11]
#     L_MedDeltoid_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,12]
#     R_Deltoid_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,13]
#     L_Deltoid_muscle[:,i] = EMG_data.iloc[Release_time-5000:Release_time+1000,14]

# Before 0.5, 1, 2 second
Before_05 =  np.zeros([np.shape(EMG_path_list)[0], 10])
Before_10 =  np.zeros([np.shape(EMG_path_list)[0], 10])
Before_20 =  np.zeros([np.shape(EMG_path_list)[0], 10])
# mean 0~0.5, 0.5~1 second
mean_0_05 = np.zeros([np.shape(EMG_path_list)[0], 10])
mean_05_1 = np.zeros([np.shape(EMG_path_list)[0], 10])


for ii in range(np.shape(EMG_path_list)[0]):    
    EMG_data = pd.read_excel(EMG_data_folder + '\\' + EMG_path_list[ii])
    Release_time = Release_time_data.iloc[ii, 1]
    print(Release_time)
    # add header    

    # extract data from data
    Before_05[ii,:] = EMG_data.iloc[Release_time-500, 1:]
    Before_10[ii,:] = EMG_data.iloc[Release_time-1000, 1:]
    Before_20[ii,:] = EMG_data.iloc[Release_time-2000, 1:]
    # calculate mean from
    mean_0_05[ii,:] = np.mean(EMG_data.iloc[Release_time-1000:Release_time, 1:])
    mean_05_1[ii,:] = np.mean(EMG_data.iloc[Release_time-2000:Release_time-1000, 1:])

Before_05 = pd.DataFrame(Before_05, columns = EMG_data.columns[1:])
Before_10 = pd.DataFrame(Before_10, columns = EMG_data.columns[1:])
Before_20 = pd.DataFrame(Before_20, columns = EMG_data.columns[1:])
mean_0_05 = pd.DataFrame(mean_0_05, columns = EMG_data.columns[1:])
mean_05_1 = pd.DataFrame(mean_05_1, columns = EMG_data.columns[1:])    
# label_first = ['1','3','4','5','6','7']
# label_second = ['1','2']

# # 顯示輸入中文
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# plt.rcParams['axes.unicode_minus'] = False

# # plot TrapeziusUpper_muscle
# plt.figure(1)  
# plt.plot(np.linspace(0,3,6000), TrapeziusUpper_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('上斜方肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusUpper.jpg'), dpi =300)

# plt.figure(2)  
# plt.plot(np.linspace(0,3,6000), TrapeziusUpper_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('上斜方肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusUpper_30second.jpg'), dpi =300)

# plt.figure(3)
# T = np.transpose(TrapeziusUpper_muscle)
# T = spm1d.util.interp(T, Q=3001)
# spm1d.plot.plot_mean_sd(T[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (上斜方肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusUpper_compare.jpg'), dpi =300)

# # -----------------Infraspinatis_muscle
# plt.figure(4)  
# plt.plot(np.linspace(0,3,6000), Infraspinatis_muscle[:, [0,1,2,3,5]], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('棘下肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Infraspinatis.jpg'), dpi =300)

# plt.figure(5)  
# plt.plot(np.linspace(0,3,6000), Infraspinatis_muscle[:, 6:9], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('棘下肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Infraspinatis_30second.jpg'), dpi =300)

# plt.figure(6)
# T2 = np.transpose(Infraspinatis_muscle)
# T2 = spm1d.util.interp(T2, Q=3001)
# spm1d.plot.plot_mean_sd(T2[[0,1,2,3,5], :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T2[6:9], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (棘下肌)' )
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'Infraspinatis_compare.jpg'), dpi =300)

# # LatissimusDorsi_muscle
# plt.figure(7)  
# plt.plot(np.linspace(0,3,6000), LatissimusDorsi_muscle[:, [0,1,2,3]], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右擴背肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'LatissimusDorsi.jpg'), dpi =300)

# plt.figure(8)  
# plt.plot(np.linspace(0,3,6000), LatissimusDorsi_muscle[:, 8], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右擴背肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'LatissimusDorsi_30second.jpg'), dpi =300)

# plt.figure(9)
# T3 = np.transpose(LatissimusDorsi_muscle)
# T3 = spm1d.util.interp(T3, Q=3001)
# spm1d.plot.plot_mean_sd(T3[[0,1,2,3], :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# # spm1d.plot.plot_mean_sd(T3[8], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.plot(np.linspace(0,3001,6000), LatissimusDorsi_muscle[:, 8], label=label_second)
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右擴背肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'LatissimusDorsi_compare.jpg'), dpi =300)

# # Biceps_muscle
# plt.figure(10)  
# plt.plot(np.linspace(0,3,6000), Biceps_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右二頭肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Biceps.jpg'), dpi =300)

# plt.figure(11)  
# plt.plot(np.linspace(0,3,6000), Biceps_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右二頭肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Biceps_30second.jpg'), dpi =300)

# plt.figure(12)
# T4 = np.transpose(Biceps_muscle)
# T4 = spm1d.util.interp(T4, Q=3001)
# spm1d.plot.plot_mean_sd(T4[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T4[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右二頭肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'Biceps_compare.jpg'), dpi =300)

# # Triceps_muscle
# plt.figure(13)  
# plt.plot(np.linspace(0,3,6000), Triceps_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右三頭肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Triceps.jpg'), dpi =300)

# plt.figure(14)  
# plt.plot(np.linspace(0,3,6000), Triceps_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右三頭肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Triceps_30second.jpg'), dpi =300)

# plt.figure(15)
# T5 = np.transpose(Triceps_muscle)
# T5 = spm1d.util.interp(T5, Q=3001)
# spm1d.plot.plot_mean_sd(T5[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T5[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右三頭肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'Triceps_compare.jpg'), dpi =300)

# # Extensor_muscle
# plt.figure(16)  
# plt.plot(np.linspace(0,3,6000), Extensor_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右伸腕肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Extensor.jpg'), dpi =300)

# plt.figure(17)  
# plt.plot(np.linspace(0,3,6000), Extensor_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右伸腕肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Extensor_30second.jpg'), dpi =300)

# plt.figure(18)
# T6 = np.transpose(Extensor_muscle)
# T6 = spm1d.util.interp(T6, Q=3001)
# spm1d.plot.plot_mean_sd(T6[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T6[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右伸腕肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'Extensor_compare.jpg'), dpi =300)

# # Flexor_muscle
# plt.figure(19)  
# plt.plot(np.linspace(0,3,6000), Flexor_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右屈腕肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Flexor.jpg'), dpi =300)

# plt.figure(20)  
# plt.plot(np.linspace(0,3,6000), Flexor_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右屈腕肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'Flexor_30second.jpg'), dpi =300)

# plt.figure(21)
# T7 = np.transpose(Flexor_muscle)
# T7 = spm1d.util.interp(T7, Q=3001)
# spm1d.plot.plot_mean_sd(T7[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T7[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右屈腕肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'Flexor_compare.jpg'), dpi =300)

# # TrapeziusLower_muscle
# plt.figure(22)  
# plt.plot(np.linspace(0,3,6000), TrapeziusLower_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右下斜方肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusLower.jpg'), dpi =300)

# plt.figure(23)  
# plt.plot(np.linspace(0,3,6000), TrapeziusLower_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右下斜方肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusLower_30second.jpg'), dpi =300)

# plt.figure(24)
# T8 = np.transpose(TrapeziusLower_muscle)
# T8 = spm1d.util.interp(T8, Q=3001)
# spm1d.plot.plot_mean_sd(T8[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T8[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右下斜方肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'TrapeziusLower_compare.jpg'), dpi =300)

# # L_TrapeziusUpper_muscle
# plt.figure(25)  
# plt.plot(np.linspace(0,3,6000), L_TrapeziusUpper_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('左上斜方肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'L_TrapeziusUpper.jpg'), dpi =300)

# plt.figure(26)  
# plt.plot(np.linspace(0,3,6000), L_TrapeziusUpper_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('左上斜方肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'L_TrapeziusUpper_30second.jpg'), dpi =300)

# plt.figure(27)
# T_11 = np.transpose(L_TrapeziusUpper_muscle)
# T_11 = spm1d.util.interp(T_11, Q=3001)
# spm1d.plot.plot_mean_sd(T_11[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T_11[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (左上斜方肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'L_TrapeziusUpper_compare.jpg'), dpi =300)

# # L_Biceps_muscle
# # plt.figure(28)  
# # plt.plot(np.linspace(0,3,6000), L_Biceps_muscle[:, 0:7], label=label_first)
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Muscle activation(%)')
# # plt.title('左二頭肌' )
# # plt.grid()
# # plt.legend()
# # plt.axvline(x=2.5, c='r', ls='--', lw=1)
# # plt.savefig((Save_figuare + '\\'+ 'L_Biceps.jpg'), dpi =300)

# # plt.figure(29)  
# # plt.plot(np.linspace(0,3,6000), L_Biceps_muscle[:, 7:10], label=label_second)
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Muscle activation(%)')
# # plt.title('左二頭肌 (30秒疲勞後)')
# # plt.grid()
# # plt.legend()
# # plt.axvline(x=2.5, c='r', ls='--', lw=1)
# # plt.savefig((Save_figuare + '\\'+ 'L_Biceps_30second.jpg'), dpi =300)

# # plt.figure(30)
# # T_12 = np.transpose(L_Biceps_muscle)
# # T_12 = spm1d.util.interp(T_12, Q=3001)
# # spm1d.plot.plot_mean_sd(T_12[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# # spm1d.plot.plot_mean_sd(T_12[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# # plt.axvline(x=2500, c='r', ls='--', lw=1)
# # plt.title('疲勞前後比較 (左二頭肌)' )
# # plt.legend()
# # plt.grid()
# # plt.savefig((Save_figuare + '\\'+ 'L_Biceps_compare.jpg'), dpi =300)

# # L_Extensor_muscle
# # plt.figure(31)  
# # plt.plot(np.linspace(0,3,6000), L_Extensor_muscle[:, 0:7], label=label_first)
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Muscle activation(%)')
# # plt.title('左伸腕肌' )
# # plt.grid()
# # plt.legend()
# # plt.axvline(x=2.5, c='r', ls='--', lw=1)
# # plt.savefig((Save_figuare + '\\'+ 'L_Extensor.jpg'), dpi =300)

# # plt.figure(32)  
# # plt.plot(np.linspace(0,3,6000), L_Extensor_muscle[:, 7:10], label=label_second)
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Muscle activation(%)')
# # plt.title('左伸腕肌 (30秒疲勞後)')
# # plt.grid()
# # plt.legend()
# # plt.axvline(x=2.5, c='r', ls='--', lw=1)
# # plt.savefig((Save_figuare + '\\'+ 'L_Extensor_30second.jpg'), dpi =300)

# # plt.figure(33)
# # T_13 = np.transpose(L_Extensor_muscle)
# # T_13 = spm1d.util.interp(T_13, Q=3001)
# # spm1d.plot.plot_mean_sd(T_13[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# # spm1d.plot.plot_mean_sd(T_13[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# # plt.axvline(x=2500, c='r', ls='--', lw=1)
# # plt.title('疲勞前後比較 (左上伸腕肌)' )
# # plt.legend()
# # plt.grid()
# # plt.savefig((Save_figuare + '\\'+ 'L_Extensor_compare.jpg'), dpi =300)

# # L_MedDeltoid_muscle
# # plt.figure(34)  
# # plt.plot(np.linspace(0,3,6000), L_MedDeltoid_muscle[:, 0:7], label=label_first)
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Muscle activation(%)')
# # plt.title('左中三角肌' )
# # plt.grid()
# # plt.legend()
# # plt.axvline(x=2.5, c='r', ls='--', lw=1)
# # plt.savefig((Save_figuare + '\\'+ 'L_MedDeltoid.jpg'), dpi =300)

# # plt.figure(35)  
# # plt.plot(np.linspace(0,3,6000), L_MedDeltoid_muscle[:, 7:10], label=label_second)
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('Muscle activation(%)')
# # plt.title('左中三角肌 (30秒疲勞後)')
# # plt.grid()
# # plt.legend()
# # plt.axvline(x=2.5, c='r', ls='--', lw=1)
# # plt.savefig((Save_figuare + '\\'+ 'L_MedDeltoid_30second.jpg'), dpi =300)

# # plt.figure(36)
# # T_14 = np.transpose(L_MedDeltoid_muscle)
# # T_14 = spm1d.util.interp(T_14, Q=3001)
# # spm1d.plot.plot_mean_sd(T_14[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# # spm1d.plot.plot_mean_sd(T_14[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# # plt.axvline(x=2500, c='r', ls='--', lw=1)
# # plt.title('疲勞前後比較 (左中三角肌)' )
# # plt.legend()
# # plt.grid()
# # plt.savefig((Save_figuare + '\\'+ 'L_MedDeltoid_compare.jpg'), dpi =300)

# # R_Deltoid_muscle
# plt.figure(37)  
# plt.plot(np.linspace(0,3,6000), R_Deltoid_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右後三角肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'R_Deltoid.jpg'), dpi =300)

# plt.figure(38)  
# plt.plot(np.linspace(0,3,6000), R_Deltoid_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('右後三角肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'R_Deltoid_30second.jpg'), dpi =300)


# plt.figure(39)
# T9 = np.transpose(R_Deltoid_muscle)
# T9 = spm1d.util.interp(T9, Q=3001)
# spm1d.plot.plot_mean_sd(T9[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T9[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (右後三角肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'R_Deltoid_compare.jpg'), dpi =300)

# # L_Deltoid_muscle
# plt.figure(40)  
# plt.plot(np.linspace(0,3,6000), L_Deltoid_muscle[:, 0:7], label=label_first)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('左後三角肌' )
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'L_Deltoid.jpg'), dpi =300)

# plt.figure(41)  
# plt.plot(np.linspace(0,3,6000), L_Deltoid_muscle[:, 7:10], label=label_second)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Muscle activation(%)')
# plt.title('左後三角肌 (30秒疲勞後)')
# plt.grid()
# plt.legend()
# plt.axvline(x=2.5, c='r', ls='--', lw=1)
# plt.savefig((Save_figuare + '\\'+ 'L_Deltoid_30second.jpg'), dpi =300)

# plt.figure(42)
# T10 = np.transpose(L_Deltoid_muscle)
# T10 = spm1d.util.interp(T10, Q=3001)
# spm1d.plot.plot_mean_sd(T10[0:7, :], linecolor='b', facecolor=(0.7,0.7,1), edgecolor='b', label='First6')
# spm1d.plot.plot_mean_sd(T10[7:10], linecolor='r', facecolor=(1,0.7,0.7), edgecolor='r', label='Second2')
# plt.axvline(x=2500, c='r', ls='--', lw=1)
# plt.title('疲勞前後比較 (左後三角肌)' )
# plt.legend()
# plt.grid()
# plt.savefig((Save_figuare + '\\'+ 'L_Deltoid_compare.jpg'), dpi =300)
# # ---------------precedure------------------

# # data_save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing'
# # Folder_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\RawData'

# # File_list = Read_File(Folder_path, True) 

# # for i in File_list:
# #     bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(i)
# #     Excel_writting(i, data_save_path, lowpass_filtered_data)

# # toc = time.process_time()
# # print("total time consume:",toc-tic)

# # tic = time.process_time()

# # MVC_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S2\Processing\AfterFilting\MVC'
# # MVC_save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S2\Processing'
# # Find_MVC_max(MVC_folder, MVC_save_path)

# # toc = time.process_time()
# # print("MVC time consume:",toc-tic)


# ## to calculate muscel activation of each muscle
# # tic = time.process_time()
# # MVC_file = r'D:\NTSU\TenLab\Archery\Archery_20220225\S2\Processing\S1_MVC.xlsx'
# # shooting_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S2\Processing\AfterFilting\Shooting'
# # fatigue_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S2\Processing\AfterFilting\Fatigue'
# # save_file_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S2\Processing\iMVC'    
# # iMVC_calculate(MVC_file, shooting_folder, fatigue_folder, save_file_path)    
# # toc = time.process_time()
# # print("MVC time consume:",toc-tic)    

# # tic = time.process_time()
# # folder_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S16\RawData\Shooting'
# # save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S16\Processing'
# # x= find_release_time(folder_path, save_path)
# # toc = time.process_time()
# # print("find peak time consume:",toc-tic)  
# # #--------------Compute and Plot FFT------------------------------------
# #

# # # frequency only half of sampling rate, due to complex part of numerical
# # N = N2 #truncate array to the last power of 2
# # xf = np.linspace(0.0, np.ceil(1.0/(2.0*T)), N//2)
# # pyfftw.forget_wisdom()
# # fft_obj = pyfftw.builders.rfft(x.values)
# # print(fft_obj) 
# # # due to our data type is series, therefore we need to extract value in the series
# # yf = fft(x.values)
# # yyf = pyfftw.FFTW(x.values)
# # # plot the figure
# # plt.figure(5)
# # # normalize
# # plt.plot(xf, 2.0/N * np.abs(yf[0:np.int(N/2)]), linewidth = 1, label = 'fft')
# # plt.plot(xf, 2.0/N * np.abs(yyf[0:np.int(N/2)]), linewidth = 1, label = 'pyfftw')
# # plt.grid()
# # plt.xlabel('Frequency (Hz)')
# # plt.ylabel('Accel (g)')
# # plt.title('FFT - ' + file_path)
# # toc = time.process_time()
# # print("FFT Time:",toc-tic)
# # plt.show()