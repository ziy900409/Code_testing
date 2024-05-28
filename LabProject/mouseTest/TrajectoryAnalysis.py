"""
分析滑鼠移動軌跡

程式架構:
1. 先製作 trace file
https://www.yorku.ca/mack/FittsLawSoftware/doc/FittsTrace.html
2. sd2 Output File: sequence-by-sequence
https://www.yorku.ca/mack/FittsLawSoftware/doc/FittsTask.html

計算參數:
tracer file:
     participant code
     condition code
     block code
     sequence - sequence number
     A - target amplitude (diameter of the layout circle in pixels)
     W - target width (diameter of target circle in pixels)
     trial - trial number within the sequence
     from_x - x coordinate of beginning of trial
     from_y - y coordinate of beginning of trial
     to_x - centre x coordinate of target
     to_y - centre y coordinate of target
     identifier ("t=", "x=", or "y=")

@author: Hsin.YH.Yang, written by May 02 2024
"""

import numpy as np
import pandas as pd
import json
import math
from statsmodels.stats.diagnostic import lilliefors # 進行常態分佈檢定
import sys
sys.path.append(r"D:\BenQ_Project\git\Code_testing\LabProject\mouseTest ")
import Analysis_function_v1 as func


# %% 初始參數定義
raw_data = pd.read_csv(r"D:\BenQ_Project\FittsDragDropTest\mouse_data.csv")
select_cir_radius_ratio = 0.6

# %%設定儲存格式

"""
purpose: 找出所有 trail by trail 的順序及路徑

1. 先利用 event 區分資料
2. 
"""
raw_data['Event'].duplicated().sum()

duplicate_info = {}

# 找出每個欄位的重複值及其數量
for column in raw_data.columns:
    duplicated_values = raw_data[column][raw_data[column].duplicated()]
    if not duplicated_values.empty:
        duplicate_counts = duplicated_values.value_counts().to_dict()
        duplicate_info[column] = duplicate_counts

# 創建 sd1 表格包含 Amplitudes * Width * Trial, 格式為 pandas.DataFrame
sd1_table = pd.DataFrame({},
                         columns = ["Participant", "Condition", "Block",
                                    "Trial", "A", "W", "Ae", "dx", "PT(ms)", "ST(ms)", "MT(ms)",
                                    "Errors", "TRE", "TAC", "MDC", "ODC", "MV", "ME", "MO"]
                         )
# 創建 sd2 表格包含 Amplitudes * Width
sd2_table = pd.DataFrame({},
                         columns = ["Participant", "Condition", "Block",
                                    "Trial", "A", "W", "Ae", "We", "IDe(bits)", "PT(ms)",
                                    "ST(ms)", "MT(ms)", "ER(%)", "TP(bps)", "TRE", "TAC", "MDC",
                                    "ODC", "MV", "ME", "MO"]
                         )
# 創建 sd3 表格包含 Amplitudes * Width * Trial, 格式為 Json
sd3_data_format = {
                  "info": {
                      "participants":[],
                      "condition": [],
                      "blocks": []
                      },
                  "task":{
                      "Sequence":[],
                      "A": [],
                      "W": [],
                      "Trial": [],
                      "from_x": [],
                      "from_y": [],
                      "to_x": [],
                      "to_y": [],
                      "select_x": [],
                      "select_y": [],
                      "MT": [],
                      "Errors": []
                      },
                  "{t_x_y}":{
                      "t":[],
                      "x": [],
                      "y": []
                      }
                  }

# 填入任務參數
# 將info.keys轉成list的數值，並依照數值大小作排列
trial_info = sorted([int(key) for key in duplicate_info['Trial'].keys()])
trial_info.remove(14)
amplitude_info = sorted([int(key) for key in duplicate_info["Amplitudes"].keys()])
width_info = sorted([int(key) for key in duplicate_info["Width"].keys()])

# %%
# 找出每筆 trial, 數量為 Amplitudes * Width * Trial, 並依照 time 作排列
for i in range(len(duplicate_info["Amplitudes"])):
    for ii in range(len(duplicate_info["Width"])):
        for iii in range(len(duplicate_info['Trial'])-1):
            print(amplitude_info[i], width_info[ii], trial_info[iii])
            # 1. 找出索引 -----------------------------------------------------
            # 找出每次任務的滑鼠軌跡
            condition = (
                        (raw_data['Event'] == 'MOUSEPOS') &
                        (raw_data['Amplitudes'] == amplitude_info[i]) &
                        (raw_data['Width'] == width_info[ii]) &
                        (raw_data['Trial'] == trial_info[iii])
                        )
            matched_indices = raw_data.index[condition].tolist()
            # 找出目標物的圓心
            target_cond = (
                            (raw_data['Event'] == 'EdgeCirclePos') &
                            (raw_data['Amplitudes'] == amplitude_info[i]) &
                            (raw_data['Width'] == width_info[ii]) &
                            (raw_data['Trial'] == trial_info[iii])
                            )
            target_indices = raw_data.index[target_cond].tolist()
            # 找出每次滑鼠按壓的索引
            buttomDown = (
                            (raw_data['Event'] == 'MOUSEBUTTONDOWN') &
                            (raw_data['Amplitudes'] == amplitude_info[i]) &
                            (raw_data['Width'] == width_info[ii]) &
                            (raw_data['Trial'] == trial_info[iii])
                            )
            buttomDown_indices = raw_data.index[buttomDown].tolist()
            # 計算 movement time: 按鍵按壓開始 -> 按鍵釋放
            # 如果沒有記錄到按鍵按壓的時間，則以 trial 的第一個 frame 做開始時間
            if len(buttomDown_indices) == 0:
                begin_time = raw_data["time"][matched_indices[0]]
            else:
                begin_time  = raw_data["time"][buttomDown_indices[0]]
            # 找出所有滑鼠釋放的時間，並且判斷任務失敗與否
            # 因為 BottomUp 的成功與否是對應下一個 trial，所以 trial_info[iii] + 1
            buttomUp = (
                        ((raw_data['Event'] == 'MOUSEBUTTONUP_SUCC') | \
                         (raw_data['Event'] == 'MOUSEBUTTONUP_FAIL')) &
                         (raw_data['Amplitudes'] == amplitude_info[i]) &
                         (raw_data['Width'] == width_info[ii]) &
                         (raw_data['Trial'] == trial_info[iii] + 1)
                         )
            buttomUp_indices = raw_data.index[buttomUp].tolist()
            sd3_data_format["task"]["Errors"].append(raw_data["Event"][buttomUp_indices[0]])

            # 2. 將資料記錄在 dict --------------------------------------------
            # 更新info部分
            sd3_data_format["info"]["participants"].append(raw_data["Participant"][0])
            sd3_data_format["info"]["condition"].append(raw_data["Condition"][0])
            sd3_data_format["info"]["blocks"].append(raw_data["Block"][0])

            # 更新task部分
            sd3_data_format["task"]["A"].append(raw_data["Amplitudes"][matched_indices[0]])
            sd3_data_format["task"]["W"].append(raw_data["Width"][matched_indices[0]])
            sd3_data_format["task"]["Trial"].append(raw_data["Trial"][matched_indices[0]])
            sd3_data_format["task"]["Sequence"].append(raw_data["Sequence"][matched_indices[0]])
            sd3_data_format["task"]["from_x"].append(raw_data["Pos_x"][matched_indices[0]])
            sd3_data_format["task"]["from_y"].append(raw_data["Pos_y"][matched_indices[0]])
            sd3_data_format["task"]["to_x"].append(raw_data["Pos_x"][target_indices[0]])
            sd3_data_format["task"]["to_y"].append(raw_data["Pos_y"][target_indices[0]])
            sd3_data_format["task"]["select_x"].append(raw_data["Pos_x"][matched_indices[-1]])
            sd3_data_format["task"]["select_y"].append(raw_data["Pos_y"][matched_indices[-1]])
            sd3_data_format["task"]["MT"].append((raw_data["time"][matched_indices[-1]] - \
                                                  begin_time))       
            
            # 更新 {t_x_y} 部分, 每一次拖曳的軌跡
            sd3_data_format["{t_x_y}"]["t"].append(raw_data["time"][matched_indices])
            sd3_data_format["{t_x_y}"]["x"].append(raw_data["Pos_x"][matched_indices])
            sd3_data_format["{t_x_y}"]["y"].append(raw_data["Pos_y"][matched_indices])


# %% 計算 sd1 table 所需參數

for i in range(len(sd3_data_format["task"]["Trial"])):
    # 1. 基本數據定義 ---------------------------------------------------------
    # 1.1. 目標圓心
    edge_cir_x = sd3_data_format["task"]["to_x"][i] 
    edge_cir_y = sd3_data_format["task"]["to_y"][i] 
    # 1.2. 游標初始位置, 需要定義每個 trial 游標移動的位置
    pointer_pos = pd.DataFrame({"x": sd3_data_format["{t_x_y}"]["x"][i].values,
                                "y": sd3_data_format["{t_x_y}"]["y"][i].values,
                                "t": sd3_data_format["{t_x_y}"]["t"][i].values})
    
    # 2. 數據計算 -------------------------------------------------------------
    # 2.0. 判斷該次 trail 是否成功
    if sd3_data_format["task"]["Errors"][i] == 'MOUSEBUTTONUP_SUCC':
        trial_Errors = 0
    else:
        trial_Errors = 1
    # 2.1. 初始參數: a, b, c, dx, Ae
    temp_a = math.hypot((sd3_data_format["task"]["to_x"][i] - sd3_data_format["task"]["from_x"][i]),
                        (sd3_data_format["task"]["to_y"][i] - sd3_data_format["task"]["from_y"][i]))
    temp_b = math.hypot((sd3_data_format["task"]["select_x"][i] - sd3_data_format["task"]["to_x"][i]),
                        (sd3_data_format["task"]["select_y"][i] - sd3_data_format["task"]["to_y"][i]))
    temp_c = math.hypot((sd3_data_format["task"]["select_x"][i] - sd3_data_format["task"]["from_x"][i]),
                        (sd3_data_format["task"]["select_y"][i] - sd3_data_format["task"]["from_y"][i]))
    temp_dx = (temp_c * temp_c - temp_b * temp_b - temp_a * temp_a) / (2.0 * temp_a)
    if sd3_data_format["task"]["Trial"][i] == 1:
        pre_temp_dx = 0
    temp_Ae = temp_a + temp_dx + pre_temp_dx
    pre_temp_dx = (temp_c * temp_c - temp_b * temp_b - temp_a * temp_a) / (2.0 * temp_a)
    
    # 2.2. PT(ms), ST(ms), MT(ms)
    # 2.2.1. PT - pointing time (ms): 
    # 拖曳的圓圈進入邊緣圓圈後開始計算，但是必須扣除被拖曳的圓圈又離開目標圓圈的時間，
    # 因此只計算最後一次進入目標圓圈的時間
    # 方法：計算中心圓與邊界圓之間的邊界點座標，距離必須小於周圍圓的半徑
    in_edge_cir = []
    for pos in range(len(pointer_pos)):
        edge_point = func.edge_calculate(pointer_pos.loc[pos, 'x'], pointer_pos.loc[pos, 'y'], # 拖曳圓圈的圓心位置
                                    edge_cir_x, edge_cir_y, # 周圍圓圈的圓心位置
                                    sd3_data_format["task"]["W"][i]*select_cir_radius_ratio) # 拖曳圓圈的半徑
        distance = math.sqrt((edge_point[0] - edge_cir_x) ** 2 + (edge_point[1] - edge_cir_y) ** 2)
        if distance <= sd3_data_format["task"]["W"][i]:
            in_edge_cir.append(True)
        else:
            in_edge_cir.append(False)
    # 找出第一個 Fasle 的位置
    for first_False in range(len(in_edge_cir) - 1, -1, -1):
        if not in_edge_cir[first_False]:
            False_idx = first_False
            break
    PT = pointer_pos.loc[False_idx, 't'] - pointer_pos.loc[0, 't']
    # 2.2.2. selection time (ms) - the time the button is down
    ST = pointer_pos.loc[len(pointer_pos)-1, 't'] - pointer_pos.loc[False_idx, 't']
    # 2.2.3. movement time (ms) - Note: MT = PT + ST
    MT = pointer_pos.loc[len(pointer_pos)-1, 't'] - pointer_pos.loc[0, 't']
    
    # 3. 計算 Movement Variability --------------------------------------------
    # 3.1. Target Re-entry (TRE)
    # If the pointer enters the target region, leaves, 
    # then re-enters the target region, then target re-entry (TRE ) occurs. 
    TRE = func.find_true_indices_and_check_continuity(in_edge_cir)
    # 3.2. Task Axis Crossing (TAC)
    # 定義 from -> to 的連線
    from_to_line = pd.DataFrame(np.zeros([len(pointer_pos), 2]),
                                columns = ["x", "y"])
    for idx in range(len(pointer_pos)):
        slope = (sd3_data_format["task"]["to_y"][i] - sd3_data_format["task"]["from_y"][i]) / \
            (len(pointer_pos) -1)
            
        from_to_line.loc[idx, "x"] = sd3_data_format["task"]["from_x"][i] + \
            (idx)*((sd3_data_format["task"]["to_x"][i] - sd3_data_format["task"]["from_x"][i])/(len(pointer_pos)-1))
  
        from_to_line.loc[idx, "y"] = slope * (idx) + sd3_data_format["task"]["from_y"][i]

    # 找出數列中是有有數字跨過 from -> to 的連線，判定數據在連線位置的上下
    TAC_list = []
    for idx in range(len(pointer_pos)):
        # 如果在連線位置的上方，給 1
        if pointer_pos.loc[idx, "y"] >= from_to_line.loc[idx, "y"]:
            TAC_list.append(1)
        # 如果在連線位置的上方，給 -1
        else:
            TAC_list.append(-1)
    TAC = func.count_sign_changes(TAC_list)
    # 3.3. Movement Direction Change (MDC) and Orthogonal Direction Change (ODC)
    # 判斷是否有方向改變，使用速度做判斷
    vel_pointer = pointer_pos.loc[:len(pointer_pos)-2, ["x", "y"]].values - \
        pointer_pos.loc[1:, ["x", "y"]].values
    MDC_list = []
    ODC_list = []
    for idx in range(len(vel_pointer)):
        # 利用是否跨過 X 軸做判斷
        if vel_pointer[idx, 1] >= 0:
            MDC_list.append(1)
        else:
            MDC_list.append(-1)
        # 利用是否跨過 Y 軸做判斷
        if vel_pointer[idx, 0] >= 0:
            ODC_list.append(1)
        else:
            ODC_list.append(-1)
    MDC = func.count_sign_changes(MDC_list)
    ODC = func.count_sign_changes(ODC_list)
    # 3.4. Movement Variability (MV): sqrt(sum(yi-y)**2/n-1)
    # Assuming the task axis is y = 0
    MV_vector = pointer_pos.loc[:, "y"].values - from_to_line.loc[:, "y"]
    MV = np.sqrt(np.sum((MV_vector - np.mean(MV_vector))**2)/ \
                 (len(pointer_pos.loc[:, "y"]) - 1))
    # 3.5. Movement Error (ME): sum(abs(yi))/n
    ME = np.sum(abs(MV_vector))/len(pointer_pos.loc[:, "y"])
    # 3.6. Movement offset (MO) is the mean deviation of sample points from the task axis.
    MO = np.mean(MV_vector)
    
    # 4. 將任務條件及計算資料輸入至表格 ----------------------------------------
    
    # trial info
    sd1_table.loc[i, "Participant"] = sd3_data_format["info"]["participants"][i] # 受試者
    sd1_table.loc[i, "Condition"] = sd3_data_format["info"]["condition"][i] # 不同條件
    sd1_table.loc[i, "Block"] =  sd3_data_format["info"]["blocks"][i] # 第幾次測試
    sd1_table.loc[i, "A"] = sd3_data_format["task"]["A"][i]
    sd1_table.loc[i, "W"] = sd3_data_format["task"]["W"][i]
    sd1_table.loc[i, "Trial"] = sd3_data_format["task"]["Trial"][i]
    
    # calculate data
    sd1_table.loc[i, "Ae"] = temp_Ae
    sd1_table.loc[i, "dx"] = pre_temp_dx
    sd1_table.loc[i, "PT(ms)"] = PT
    sd1_table.loc[i, "ST(ms)"] = ST
    sd1_table.loc[i, "MT(ms)"] = MT
    sd1_table.loc[i, "Errors"] = trial_Errors
    
    sd1_table.loc[i, "TRE"] = TRE
    sd1_table.loc[i, "TAC"] = TAC
    sd1_table.loc[i, "MDC"] = MDC
    sd1_table.loc[i, "ODC"] = ODC
    sd1_table.loc[i, "MV"] = MV
    sd1_table.loc[i, "ME"] = ME
    sd1_table.loc[i, "MO"] = MO
    
# %% 計算 sd2 table 所需參數, 使用 sd1 table 做計算

# 0. 定義表格: 欄位名稱 xFrom, yFrom, xTo, yTo, xSelect, ySelect, MT -----------

for column in sd1_table.loc[:, ["A", "W"]]:
    duplicated_values = sd1_table[column][sd1_table[column].duplicated()]
    if not duplicated_values.empty:
        duplicate_counts = duplicated_values.value_counts().to_dict()
        duplicate_info[column] = duplicate_counts
# 定義 A, W 各自的值
A_cond = list(duplicate_info["A"].keys())
W_cond = list(duplicate_info["W"].keys())

for a_indi in range(len(A_cond)):
    for w_indi in range(len(W_cond)):
        i = a_indi*len(W_cond) + w_indi
        print(i)
        condition = (
                    (sd1_table.loc[:, "A"] == A_cond[a_indi]) &
                    (sd1_table.loc[:, "W"] == W_cond[w_indi])
                    )
        matched_indices = sd1_table.index[condition].tolist()
        # sd1_table.loc[matched_indices, "Trial"]

        # 1. 計算數據基本定義 -------------------------------------------------
        data = sd1_table.loc[matched_indices, "dx"].astype(float)
        n = len(data)
        # 2. 統計參數計算 -----------------------------------------------------
        mean_x = np.mean(sd1_table.loc[matched_indices, "dx"])
        std_x = np.std(sd1_table.loc[matched_indices, "dx"])
        # calculate kurtosis
        kurtosis = (n*(n+1) / ((n-1)*(n-2)*(n-3))) * np.sum(((data - mean_x) / std_x)**4) \
            - (3*(n-1)**2 / ((n-2)*(n-3)))
        # calculate skewness
        skewness = (n / ((n-1)*(n-2))) * np.sum(((data - mean_x) / std_x)**3)
        # 進行 Lilliefors 檢定，檢定數據是否符合常態分佈
        stat, p = lilliefors(data.values)
        # 判断是否為常態分佈
        alpha = 0.05
        if p > alpha:
            is_normal = "True"
        else:
            is_normal = "False"
        # 計算 Ae, We, IDe(bits), MT, TP
        We = 4.133 * std_x # 計算 We = 4.133 × SDx
        Ae = np.mean(sd1_table.loc[matched_indices, "Ae"]) # 計算 Ae
        IDe = math.log2(Ae/We + 1) # IDe = log2(Ae / We + 1)
        MT = np.mean(sd1_table.loc[matched_indices, "MT(ms)"])
        TP = IDe / MT * 1000 # thoughput
        # 計算 PT(ms), ST(ms) 的平均
        PT = np.mean(sd1_table.loc[matched_indices, "PT(ms)"])
        ST = np.mean(sd1_table.loc[matched_indices, "ST(ms)"])
        # 計算 Errors: 計算 MOUSEBUTTONUP_FAIL 的數量，並除以資料長度
        Errors = sd1_table.loc[matched_indices, "Errors"].value_counts()[1] / n * 100
        # 計算 TRE, TAC, MDC, ODC, MV, ME, MO 的平均值
        TRE = np.mean(sd1_table.loc[matched_indices, "TRE"])
        TAC = np.mean(sd1_table.loc[matched_indices, "TAC"])
        MDC = np.mean(sd1_table.loc[matched_indices, "MDC"])
        ODC = np.mean(sd1_table.loc[matched_indices, "ODC"])
        MV = np.mean(sd1_table.loc[matched_indices, "MV"])
        ME = np.mean(sd1_table.loc[matched_indices, "ME"])
        MO = np.mean(sd1_table.loc[matched_indices, "MO"])
        
        # 4. 將任務條件及計算資料輸入至表格 ----------------------------------------
        # trial info
        sd2_table.loc[i, "Participant"] = sd1_table.loc[matched_indices[0], "Participant"] # 受試者
        sd2_table.loc[i, "Condition"] = sd1_table.loc[matched_indices[0], "Condition"] # 不同條件
        sd2_table.loc[i, "Block"] =  sd1_table.loc[matched_indices[0], "Block"] # 第幾次測試
        sd2_table.loc[i, "A"] = sd1_table.loc[matched_indices[0], "A"]
        sd2_table.loc[i, "W"] = sd1_table.loc[matched_indices[0], "W"]
        sd2_table.loc[i, "Trial"] = sd1_table.loc[matched_indices[0], "Trial"]
        
        # calculate data
        sd2_table.loc[i, "Ae"] = Ae
        sd2_table.loc[i, "We"] = We
        sd2_table.loc[i, "IDe(bits)"] = IDe     
        sd2_table.loc[i, "PT(ms)"] = PT
        sd2_table.loc[i, "ST(ms)"] = ST
        sd2_table.loc[i, "MT(ms)"] = MT
        sd2_table.loc[i, "ER(%)"] = Errors
        sd2_table.loc[i, "TP(bps)"] = TP
        
        sd2_table.loc[i, "TRE"] = TRE
        sd2_table.loc[i, "TAC"] = TAC
        sd2_table.loc[i, "MDC"] = MDC
        sd2_table.loc[i, "ODC"] = ODC
        sd2_table.loc[i, "MV"] = MV
        sd2_table.loc[i, "ME"] = ME
        sd2_table.loc[i, "MO"] = MO

# %% sd3 輸出 csv?

            

     












