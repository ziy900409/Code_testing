# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 19:55:41 2022

@author: drink
"""
import os
import pandas as pd
import numpy as np
from scipy import signal
import time
import math

def Read_File(x, y, subfolder='None'):
    
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    data_type = y
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
        # need to change here [1:]
        for ii in file_list_1[1:]:
            file_list = os.listdir(ii)
            for iii in file_list:
                if os.path.splitext(iii)[1] == data_type:
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == data_type:
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(file_list_name)                
        
    return csv_file_list

# ---------------------EMG data processing--------------------------------
def EMG_processing(cvs_file_list):
 
    data = pd.read_csv(cvs_file_list,encoding='UTF-8')
    # to define data time
    data_time = data.iloc[:,0]
    Fs = 1/(data_time[2]-data_time[1]) # sampling frequency
    data = pd.DataFrame(data)
    # need to change column name or using column number
    EMG_data = data.iloc[:, [1, 15, 29, 43, 57, 65, 73, 81, 89, 97, 105]]
    # 1 9 17 25 33 41 49
    # exchange data type to float64
    EMG_data = [pd.to_numeric(EMG_data.iloc[:, i], errors = 'coerce') 
                for i in range(np.shape(EMG_data)[1])]
    EMG_data = pd.DataFrame(np.transpose(EMG_data),
                            columns=data.iloc[:, [1, 15, 29, 43, 57, 65, 73, 81, 89, 97, 105]].columns)
    # bandpass filter use in signal
    bandpass_sos = signal.butter(2, [20/0.802, 450/0.802],  btype='bandpass', fs=Fs, output='sos')
    
    bandpass_filtered_data = np.zeros(np.shape(EMG_data))
    for i in range(np.shape(EMG_data)[1]):
        # print(i)
        # using dual filter to processing data to avoid time delay
        bandpass_filtered = signal.sosfiltfilt(bandpass_sos, EMG_data.iloc[:,i])
        bandpass_filtered_data[:, i] = bandpass_filtered 
    
    # caculate absolute value to rectifiy EMG signal
    bandpass_filtered_data = abs(bandpass_filtered_data)     
    # -------Data smoothing. Compute RMS
    # The user should change window length and overlap length that suit for your experiment design
    # window width = window length(second)//time period(second)
    window_width = int(0.04265/(1/np.floor(Fs))) #width of the window for computing RMS
    rms_data = np.zeros(np.shape(bandpass_filtered_data))
    for i in range(np.shape(rms_data)[1]):
        for ii in range(np.shape(rms_data)[0]-window_width):
            data_location = ii+(window_width/2)
            rms_data[int(data_location), i] = (np.sum(bandpass_filtered_data[ii:ii+window_width, i])
                               /window_width)
    # ------linear envelop analysis-----------                          
    # ------lowpass filter parameter that the user must modify for your experiment        
    lowpass_sos = signal.butter(2, 6/0.802, btype='low', fs=Fs, output='sos')        
    lowpass_filtered_data = np.zeros(np.shape(bandpass_filtered_data))
    for i in range(np.shape(rms_data)[1]):
        lowpass_filtered = signal.sosfiltfilt(lowpass_sos, bandpass_filtered_data[:,i])
        lowpass_filtered_data[:, i] = lowpass_filtered
    # add columns name to data frame
    bandpass_filtered_data = pd.DataFrame(bandpass_filtered_data, columns=EMG_data.columns)
    rms_data = pd.DataFrame(rms_data, columns=EMG_data.columns)
    lowpass_filtered_data = pd.DataFrame(lowpass_filtered_data, columns=EMG_data.columns)
    # insert time data in the DataFrame
    lowpass_filtered_data.insert(0, 'time', data_time)
    rms_data.insert(0, 'time', data_time)
    bandpass_filtered_data.insert(0, 'time', data_time)    
    return bandpass_filtered_data, rms_data, lowpass_filtered_data

# ------------to find maximum MVC value-----------------------------

def Find_MVC_max(MVC_folder, MVC_save_path):
    #MVC_folder = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing\MVC'
    #MVC_save_path = r'D:\NTSU\TenLab\Archery\Archery_20220225\S1\Processing'
    MVC_file_list = os.listdir(MVC_folder)
    MVC_data = pd.read_excel(MVC_folder + '\\' + MVC_file_list[0])
    find_max_all = []
    Columns_name = MVC_data.columns
    Columns_name = Columns_name.insert(0, 'FileName')
    find_max_all = pd.DataFrame(find_max_all, columns=Columns_name)
    for i in MVC_file_list:
        MVC_file_path = MVC_folder + '\\' + i
        MVC_data = pd.read_excel(MVC_file_path)
        find_max = MVC_data.max(axis=0)
        find_max = pd.DataFrame(find_max)
        find_max = np.transpose(find_max)
        find_max.insert(0, 'FileName', i)
        find_max_all = pd.concat([find_max_all, find_max],
                                 ignore_index=True)
    # find maximum value from each file
    MVC_max = find_max_all.max(axis=0)
    MVC_max[0] = 'Max value'
    MVC_max = pd.DataFrame(MVC_max)
    MVC_max = np.transpose(MVC_max)
    find_max_all = pd.concat([find_max_all, MVC_max],
                             ignore_index=True)
    # writting data to EXCEl file
    find_max_name = MVC_save_path + '\\' + MVC_save_path.split('\\')[-2] + '_FindMVC.xlsx'
    pd.DataFrame(find_max_all).to_excel(find_max_name, sheet_name='Sheet1', index=False, header=True)

#%% code start
# # MVC path
# MVC_path = r'E:\Hsin\NTSU_lab\Lin\pick up\pick up\All in'
MVC_path = r'D:\BenQ_Project\python\LinLin\All in'
MVC_folder_list = os.listdir(MVC_path + '\\Raw_Data\\')

# 處理MVC
for MVC_folder in MVC_folder_list:
    path = MVC_path + '\\Raw_Data\\' + MVC_folder + '\\MVC'
    MVC_list = Read_File(path, '.csv', subfolder=False)
    for data_path in MVC_list:
        print(data_path)
        tic = time.process_time()
        bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(data_path)
        # 寫資料近excel
        filepath, tempfilename = os.path.split(data_path)
        filename, extension = os.path.splitext(tempfilename)
        # rewrite file name
        file_name = MVC_path + '\\Processing_Data\\' + MVC_folder + '\\MVC\\AfterFiltered\\' + filename + '_lowpass' + '.xlsx'
        # writting data in worksheet
        pd.DataFrame(lowpass_filtered_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
        toc = time.process_time()
        print("Total Time:",toc-tic)
    Find_path = MVC_path + '\\Processing_Data\\' + MVC_folder + '\\MVC\\AfterFiltered'
    MVC_save_path = MVC_path + '\\Processing_Data\\' + MVC_folder + "\\MVC"
    Find_MVC_max(Find_path, MVC_save_path)
    
# %%
# 濾波EMG of motion，並擷取特定區段
# Save_motion_path = r'E:\Hsin\NTSU_lab\Lin\pick up\pick up\All in\Processing_Data'
# motion_folder = r"E:\Hsin\NTSU_lab\Lin\pick up\pick up\All in\Raw_Data"

Save_motion_path = r'D:\BenQ_Project\python\LinLin\All in\2.Processing_Data'
motion_folder = r"D:\BenQ_Project\python\LinLin\All in\1.Raw_Data"


motion_folder_list = os.listdir(motion_folder)
# staging data
# staging_data = pd.read_excel(r"E:\Hsin\NTSU_lab\Lin\pick up\pick up\Tennis_Staging_3m_Lin_20221017.xlsx")
staging_data = pd.read_excel("D:\BenQ_Project\python\LinLin\Tennis_Staging_3m_Lin_20221017.xlsx")

for folder in motion_folder_list:
    motion_folder_path = motion_folder + '\\' + folder + "\\motion"
    motion_file_list = Read_File(motion_folder_path, '.csv', subfolder=False)
    # 讀取MVC數據
    MVC_path = Save_motion_path + '\\' + folder + '\\MVC\\' + folder + '_FindMVC.xlsx'
            
    MVC_value = pd.read_excel(MVC_path)
    MVC_value = MVC_value.iloc[-1, 1:]
    MVC_value = MVC_value.iloc[1:]
    for path in range(len(motion_file_list)):
        for num in range(len(staging_data['EMG_file_name'])):
            if type(staging_data['EMG_file_name'][num]) == str:
                if motion_file_list[path].split('\\')[-1] == staging_data['EMG_file_name'][num].split('\\')[-1]:
                    # tic2 = time.process_time()
                    print(motion_file_list[path])
                    print(staging_data['EMG_file_name'][num])
                    bandpass_filtered_data, rms_data, lowpass_filtered_data = EMG_processing(motion_file_list[path])
                    # 寫入時間
                    # time_data = pd.read_csv(motion_file_list[path], encoding='UTF-8')
                    # time = time_data.iloc[:, 0]
                    # lowpass insert time
                    # 寫資料進excel，定義檔名
                    filepath, tempfilename = os.path.split(motion_file_list[path])
                    filename, extension = os.path.splitext(tempfilename)
                    # 計算標準化EMG
                    # 設立儲存位置
                    shooting_iMVC = pd.DataFrame(np.zeros(np.shape(bandpass_filtered_data)),
                                                 columns = bandpass_filtered_data.columns)
                    shooting_iMVC.iloc[:, 1:] = abs(np.divide(bandpass_filtered_data.iloc[:, 1:], MVC_value)*100)
                    shooting_iMVC.iloc[:, 0] = bandpass_filtered_data.iloc[:, 0]
                    # 將資料寫進excel
                    # filepath, tempfilename = os.path.split(iii)
                    save_iMVC_name = Save_motion_path + '\\' + folder + '\\motion\\iMVC\\'  + filename + '_iMVC.xlsx'
                    
                    
                    # rewrite file name
                    save_lowpass_name = Save_motion_path + '\\' + folder + '\\motion\\AfterFiltered\\' + filename + '_lowpass.xlsx'
                    if math.isnan(staging_data['進力版時間1'][num]) != True:
                        # writting data in worksheet
                        start_time = int((staging_data['進力版時間1'][num] - staging_data['trigger時間'][num]) / 2400 * 2000)
                        end_time = int((staging_data['出力版時間2'][num]- staging_data['trigger時間'][num]) / 2400 * 2000)
                        print('starting time is: ', start_time)
                        print('ending time is: ', end_time)
                        # write EMG data of lowpass data
                        # trunkcate specific period
                        shooting_data = lowpass_filtered_data.iloc[start_time:end_time, :]
                        pd.DataFrame(shooting_data).to_excel(save_lowpass_name,
                                                             index=False, header=True)
                        # write EMG data of iMVC data
                        pd.DataFrame(shooting_iMVC).to_excel(save_iMVC_name,
                                                             index=False, header=True)
                # toc2 = time.process_time()
                # print("Total Time:",toc2-tic2)

# %%
# 設定資料夾位置
# MVC_max_path = r'D:\2022Tennis data\3MEMG+FP\All in\MVC'                    
# motion_folder_path = r'E:\Hsin\NTSU_lab\Lin\pick up\pick up\All in\2.Processing_Data'
# save_file_path = r'D:\2022Tennis data\3MEMG+FP\All in\processing_motion'

motion_folder_path = r'D:\BenQ_Project\python\LinLin\All in\2.Processing_Data'
save_file_path = r'D:\BenQ_Project\python\LinLin\All in\3.Statistic'
# 資料夾清單
# MVC_max_folder_list = os.listdir(MVC_max_path)
motion_folder_list = os.listdir(motion_folder_path)


period1_mean_ForcePlate_data = pd.DataFrame({
                                    })
period2_mean_ForcePlate_data = pd.DataFrame({
                                    })
period3_mean_ForcePlate_data = pd.DataFrame({
                                    })
period4_mean_ForcePlate_data = pd.DataFrame({
                                    })

for i in range(len(motion_folder_list)):
    print(motion_folder_list[i])
    motion_path = motion_folder_path + '\\' + motion_folder_list[i] + '\\motion\\iMVC'
    motion_list = Read_File(motion_path,
                            '.xlsx', subfolder=False)
    for iii in motion_list:
        filepath, tempfilename = os.path.split(iii)
        filename, _ = os.path.splitext(tempfilename)
        Period1_add_mean_motion = pd.DataFrame({
            })
        Period2_add_mean_motion = pd.DataFrame({
            })
        Period3_add_mean_motion = pd.DataFrame({
            })
        Period4_add_mean_motion = pd.DataFrame({
            })
        for num in range(len(staging_data['low_file_EMG'])):
            if type(staging_data['EMG_file_name'][num]) == str:
                stage_file, _ = os.path.splitext(staging_data['EMG_file_name'][num].split('\\')[-1])
                if stage_file in filename:
                    if np.isnan(staging_data['進力版時間1'][num]) != 1:
                        print(iii)
                        print(MVC_path)
                        print(staging_data['low_file_EMG'][num])
                        # 定義分期時間
                        start_time1 = 0
                        end_time1 = int((staging_data['出力版時間1'][num] - staging_data['進力版時間1'][num]) / 2400 * 2000)
                        start_time2 = int((staging_data['進力版時間2'][num] - staging_data['進力版時間1'][num]) / 2400 * 2000)
                        end_time2 = int((staging_data['出力版時間2'][num]- staging_data['進力版時間1'][num]) / 2400 * 2000)
                        # 讀取資料
                        motion_data =  pd.read_excel(iii)
                        # 找最大值
                        # max_motion = pd.DataFrame(np.max(shooting_iMVC, axis=0))
                        # max_motion = np.transpose(max_motion)
                        file_name = pd.DataFrame({
                            'file_name': [iii]
                            })
                        # add_max_motion = pd.concat([file_name, max_motion], ignore_index=True, axis= 1)
                        # 找平均值
                        
                        
                        # period 1
                        period11_mean_motion = pd.DataFrame([motion_data.iloc[start_time1:int(end_time1/2), :].mean()],
                                                           columns = motion_data.columns)
                        period11_mean_motion.insert(0, 'task', 'period1-1 mean')
                        period11_mean_motion.insert(1, 'trial', filename)
                        
                        period12_mean_motion = pd.DataFrame([motion_data.iloc[int(end_time1/2):end_time1, :].mean()],
                                                           columns = motion_data.columns)
                        period12_mean_motion.insert(0, 'task', 'period1-2 mean')
                        period12_mean_motion.insert(1, 'trial', filename)
                        
                        
                        # period 2
                        period21_mean_motion = pd.DataFrame([motion_data.iloc[start_time2:int(end_time2/2), :].mean()],
                                                           columns = motion_data.columns)
                        period21_mean_motion.insert(0, 'task', 'period2-1 mean')
                        period21_mean_motion.insert(1, 'trial', filename)
                        
                        period22_mean_motion = pd.DataFrame([motion_data.iloc[int(end_time2/2):end_time2, :].mean()],
                                                           columns = motion_data.columns)
                        period22_mean_motion.insert(0, 'task', 'period2-2 mean')
                        period22_mean_motion.insert(1, 'trial', filename)
                        
                        
                        # period all
                        # period31_mean_motion = pd.DataFrame([motion_data.iloc[start_time1:int(end_time2/2), :].mean()],
                        #                                    columns = motion_data.columns)
                        # period31_mean_motion.insert(0, 'task', 'period3-1 mean')
                        # period31_mean_motion.insert(1, 'trial', filename)
                        
                        # period32_mean_motion = pd.DataFrame([motion_data.iloc[int(end_time2/2):end_time2, :].mean()],
                        #                                    columns = motion_data.columns)
                        # period32_mean_motion.insert(0, 'task', 'period3-2 mean')
                        # period32_mean_motion.insert(1, 'trial', filename)
      

                        # 資料貯存矩陣
                        
                        # 合併計算資料
                        add_emg_statics = pd.concat([period11_mean_motion, period12_mean_motion,
                                                     period21_mean_motion, period22_mean_motion
                                                     ])
                    
                        
            # 合併資料
            # all_max_ForcePlate_data = pd.concat([all_max_ForcePlate_data, add_max_motion], ignore_index=True, axis= 0)
            period1_mean_ForcePlate_data = pd.concat([period1_mean_ForcePlate_data, Period1_add_mean_motion], ignore_index=True, axis= 0)
            period2_mean_ForcePlate_data = pd.concat([period2_mean_ForcePlate_data, Period2_add_mean_motion], ignore_index=True, axis= 0)
            period3_mean_ForcePlate_data = pd.concat([period3_mean_ForcePlate_data, Period3_add_mean_motion], ignore_index=True, axis= 0)
            period4_mean_ForcePlate_data = pd.concat([period4_mean_ForcePlate_data, Period4_add_mean_motion], ignore_index=True, axis= 0)

# excel_save = r'D:\2022Tennis data\3MEMG+FP\\'
# with pd.ExcelWriter((excel_save + '3m_EMG.xlsx')) as writer:
#     pd.DataFrame(period1_mean_ForcePlate_data).to_excel(writer, sheet_name='time1', index=False, header=True) 
#     pd.DataFrame(period2_mean_ForcePlate_data).to_excel(writer, sheet_name='time2', index=False, header=True) 
#     pd.DataFrame(period3_mean_ForcePlate_data).to_excel(writer, sheet_name='time3', index=False, header=True)
#     pd.DataFrame(period4_mean_ForcePlate_data).to_excel(writer, sheet_name='time4', index=False, header=True)    