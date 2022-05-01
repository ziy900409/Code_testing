import os
import pandas as pd
import numpy as np
from pandas import DataFrame

# 自動讀檔用，可判斷路徑下所有檔名，不管有沒有子資料夾
# 可針對不同副檔名的資料作判讀
def Read_File(x, subfolder='None'):
    # if subfolder = True, the function will run with subfolder
    folder_path = x
    csv_file_list = []
    
    if subfolder:
        file_list_1 = []
        for dirPath, dirNames, fileNames in os.walk(x):
            # file_list = os.walk(folder_name)
            file_list_1.append(dirPath)
                
        for ii in file_list_1:
            file_list = os.listdir(ii)
            for iii in file_list:
                # 抓副檔名.trc的檔案，可修正為抓多種不同副檔名
                if os.path.splitext(iii)[1] == ".trc":
                    file_list_name = ii + '\\' + iii
                    csv_file_list.append(file_list_name)
    else:
        folder_list = os.listdir(x)                
        for i in folder_list:
            if os.path.splitext(i)[1] == ".trc":
                # 抓副檔名.trc的檔案
                file_list_name = folder_path + "\\" + i
                csv_file_list.append(i)                
        
    return csv_file_list

# -------------------------code staring---------------------------------

# read staging file
# 設定分期檔的路徑
staging_file_path = r'D:\NTSU\TenLab\ComputerMouse\S3_Frame_TeskRecord.xlsx'
staging_file_data = pd.read_excel(staging_file_path, sheet_name="large range")
# read data
# 設定範例資料檔用，可用隨意一種檔案做使用
data_path = r'D:\NTSU\TenLab\ComputerMouse\data\S1_L_lifting_1_1-Mouse 1.trc'
example_data = pd.read_csv(data_path, delimiter='\t' ,skiprows=3, encoding='UTF-8')

# read file list
# setting file's folder
# 設定目標資料夾
folder_path = r'D:\NTSU\TenLab\ComputerMouse\data'
file_list = Read_File(folder_path, subfolder=True)

# Determine the file name
# 預先創建貯存資料的矩陣
calculate_data = np.zeros([np.shape(file_list)[0], np.shape(example_data)[1]])
# 設定欄位名稱
columns_name = example_data.columns[0:-1]
columns_name = columns_name.insert(0, 'FileName')
calculate_data = pd.DataFrame(calculate_data, columns=columns_name)
for i in range(len(file_list)):
    for ii in range(len(staging_file_data['FileName'])):
        if file_list[i] == staging_file_data['FileName'][ii]:
            print(i)
            print(ii)
            print(staging_file_data['FileName'][ii])
            print(file_list[i])
            # read data
            mouse_data = pd.read_csv(file_list[i], delimiter='	' ,skiprows=3, encoding='UTF-8')
            # using staging file to extract data
            # 利用分期檔抓時間點
            start_frame = staging_file_data['Start Frame'][ii]
            end_frame = staging_file_data['End Frame'][ii]
            extract_data = mouse_data.iloc[start_frame:end_frame+1, :-1]
            # convert str to float data type
            # 轉換資料格式 從字串變浮點數
            extract_data.iloc[:, 2:] = extract_data.iloc[:, 2:].astype(float)
            # calculate mean value
            # 計算平均值
            mean_extract_data = np.average(extract_data, axis=0)
            # assign data to claculate matrix
            calculate_data.iloc[i, 0] = file_list[i]
            calculate_data.iloc[i, 1:] = mean_extract_data
# write data to excel
# 將資料寫進EXCEL，可修改檔案名稱
file_name = r'D:\NTSU\TenLab\test\output\S3_output.xlsx'
DataFrame(extract_data).to_excel(file_name, sheet_name='Sheet1', index=False, header=True)
