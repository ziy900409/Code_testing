# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:55:42 2023

https://kknews.cc/code/bgxv6vo.html

@author: Hsin.YH.Yang
"""

# %% 目錄

# 1. 載入套件
# 2. 讀取檔案，存入 DataFrame (！請調整 本次分析的按鍵 為 '左鍵' or '右鍵'，以及檔名、日期)
# 3. 擷取 Rating 和 Data 資料
# 4. Rating 調整為兩類，並新增 1 總評等欄位
# 5. 設定問題與 X、y (！請填寫本次分析的問題)
# 6. GridSearch key hyperparameters with RepeatedStratifiedKFold
# 7. 輸出 bestModel 決策樹
# 8. 建立 finalTable (！只需要執行一次)
# 9. 產出 summaryTable，加至 finalTable 之後
# 10. 輸出 finalTable 成 Excel 檔 (全部資料都存入 finalTable 才執行)



# %% 1. 載入套件
import pandas as pd
import numpy as np
# import math
import time
import seaborn as sns
# from IPython import get_ipython
import gc

import matplotlib.pyplot as plt  # 視覺化
from matplotlib import rcParams  # 圖設定
rcParams['figure.figsize'] = (25, 20)  # 圖大小
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 將圖的中文字體設為微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 用來正常顯示負號

# from sklearn.tree import DecisionTreeClassifier as dtc  # 決策樹演算法
from sklearn.model_selection import train_test_split  # 拆分訓練集及測試集
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # 模型準確度
# from sklearn.tree import plot_tree  # 決策樹圖
from sklearn.model_selection import cross_val_score # 交叉驗證
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold # define the evaluation procedure
    # evaluate a given model using cross-validation
    # RepeatedStratifiedKFold: 重複執行StratifiedKFold
    # StratifiedKFold: 將樣本數"分層採樣StratifiedKFold"分為K份，
    # 每次隨機選擇K-1使用重複StratifiedKFold來拆分數據做訓練集，
    # 剩下一份作測試集，執行若干輪(>K)，在選擇損失函數評估最優的模型與參數
    # https://www.cnblogs.com/liuxiangyan/p/14299865.html..
from sklearn.model_selection import GridSearchCV #貪婪搜索最佳參數
from sklearn.inspection import permutation_importance #random forest比較特徵permutation important

# %% 測試資料
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
import warnings 
warnings.filterwarnings("ignore") 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix 
# import statsmodels.formula.api as sm
import statsmodels.api as sm

np.random.seed(123)

data = pd.read_csv(r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\data.csv")
data = data.iloc[:,1:-1]

label_encoder = LabelEncoder() 
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')

corr = data.corr()
sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool) 
for i in range(corr.shape[0]): 
    for j in range(i+1, corr.shape[0]): 
        if corr.iloc[i,j] >= 0.9 or corr.iloc[i,j] <= -0.9: 
            if columns[j]: 
                columns[j] = False
selected_columns = data.columns[columns] 
data = data[selected_columns]


selected_columns = selected_columns[1:].values 
# import statsmodels.formula.api as sm 
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0]) 
    for i in range(0, numVars): 
        regressor_OLS = sm.OLS(Y, x).fit() 
        print(regressor_OLS.summary())
        maxVar = max(regressor_OLS.pvalues).astype(float) 
        if maxVar > sl: 
            for j in range(0, numVars - i): 
                if (regressor_OLS.pvalues[j].astype(float) == maxVar): 
                    x = np.delete(x, j, 1) 
                    columns = np.delete(columns, j) 
                    regressor_OLS.summary() 
                    return x, columns 
SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:,1:].values, data.iloc[:,0].values, SL, selected_columns)



# %% 2. 讀取檔案，存入 DataFrame (！請調整 本次分析的按鍵 為 '左鍵' or '右鍵'，以及檔名、日期)

analKey = 'EVT-data'  ## 本次分析的按鍵
## 使用 Data 之檔名
inputFileName = r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\03_第二輪_決策樹\EC2\EC2-CW-key-para-collect-20230522.xlsx" 


analDate = '0524'  ## 分析日期

inputFile = pd.read_excel(inputFileName, sheet_name = analKey)  # 讀入 excel 資料
data_version = '(old 40)'  # 資料版本

fig_save_path = r"C:\Users\Public\BenQ\01_UR_lab\01_DecisionTree\第二輪_決策樹\results\\"
# %% 3. 擷取 Rating 和 Data 資料
right_key = "右_"
left_key = "左_"
# 設定評分資料的欄位名稱
right_rating_columns = []
left_rating_columns = []
for columns_name in inputFile.iloc[:, 6:18].columns:
    if right_key in columns_name:
        right_rating_columns.append(columns_name)
    elif left_key in columns_name:
        left_rating_columns.append(columns_name)
# 設定輸入資料的欄位名稱
right_input_columns = []
left_input_columns = []
for columns_name in inputFile.iloc[:, 144:].columns:
    if right_key in columns_name:
        right_input_columns.append(columns_name)
    elif left_key in columns_name:
        left_input_columns.append(columns_name)
# 設定資料
## 評分資料
right_rating = inputFile.loc[:, right_rating_columns]
left_rating = inputFile.loc[:, left_rating_columns]
## 輸入資料
right_input = inputFile.loc[:, right_input_columns]
left_input = inputFile.loc[:, left_input_columns]
# 先捨棄全為0或是Nan的欄位
right_input = right_input.dropna(axis=1)
right_input = right_input.loc[:, (right_input.apply(np.sum, axis=0) !=0).values]
left_input = left_input.dropna(axis=1)
left_input = left_input.loc[:, (left_input.apply(np.sum, axis=0) !=0).values]
# 存取結果
all_results = pd.DataFrame({}, columns = ['algorithm', 'question', 'best_params', 'accuracy','balanced_accuracy',
                                          'f1 score', 'data_version', 'analKey'])

# %% 4. Rating 調整為兩類，並新增 1 總評等欄位 (有任一 1 即為 1)
#  先將 A B 放置為一組，C 單獨一組
right_rating.insert(0, 'final_score', 0)
# for i in range(len(np.shape(right_rating)[1])):
    

right_rating = right_rating.replace(['A', 'B', 'C', 'D', 'E'], [0, 0, 1, 1, 1])
# 設定如果有任一欄位有C以上的值，就將 final score 設定為 1
right_rating.iloc[:, 0]  = (right_rating.iloc[:, 1:].apply(np.sum, axis=1) != 0).values
right_rating.iloc[:, 0] = right_rating.iloc[:, 0].replace([True, False], [1, 0])
# twoCategory.replace(['C'], 1, inplace=True)  # DE 為 1
# 先捨棄全為0或是Nan的欄位
# right_rating = right_rating.dropna(axis=1)

# 統計各欄位 0 1 數量
# categoryCounts = twoCategory.apply(pd.Series.value_counts).rename_axis('Category')

# %% 5.1. 清理資料 : 相關係數分析與劃出資料分佈圖
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
import warnings 
warnings.filterwarnings("ignore") 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix 
np.random.seed(123)

corr = right_input.corr()
fig= plt.plot(figsize=(14, 10))
sns.heatmap(corr)
# 丟棄相關係數高於0.9的參數
columns = np.full((corr.shape[0],), True, dtype=bool) 
for i in range(corr.shape[0]): 
    for j in range(i+1, corr.shape[0]): 
        if corr.iloc[i,j] >= 0.9 or corr.iloc[i,j] <= -0.9: 
            if columns[j]: 
                columns[j] = False
selected_columns = right_input.columns[columns] 
right_input = right_input[selected_columns]

selected_columns = selected_columns[1:].values 
# import statsmodels.formula.api as sm
import statsmodels.api as sm 
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0]) 
    for i in range(0, numVars): 
        regressor_OLS = sm.OLS(Y, x).fit() 
        maxVar = max(regressor_OLS.pvalues).astype(float) 
        if maxVar > sl: 
            for j in range(0, numVars - i): 
                if (regressor_OLS.pvalues[j].astype(float) == maxVar): 
                    x = np.delete(x, j, 1) 
                    columns = np.delete(columns, j) 
                    regressor_OLS.summary() 
                    return x, columns 
SL = 0.05
data_modeled, selected_columns = backwardElimination(right_input.values, right_rating.iloc[:,1].values, SL, selected_columns)
# 繪製資料分布圖
for ii in right_rating:
    print(ii)
    fig = plt.figure(figsize = (20, 25)) 
    j = 0 
    for i in right_input.columns: 
        plt.subplot(7, 4, j+1) 
        sns.distplot(right_input[i][right_rating[ii]==0], color='g', label = 'A') 
        sns.distplot(right_input[i][right_rating[ii]==1], color='r', label = 'B')
        plt.title(right_input.columns[j])
        plt.legend(loc='best')
        j += 1 
    fig.suptitle(ii, fontsize = 12) 
    fig.tight_layout() 
    fig.subplots_adjust(top=0.95) 
    plt.show()
# %% 5.2. 清理資料 : 找出離群值
## 四分位距

plt.boxplot(tips[['total_bill', 'tip']], labels=['total_bill', 'tip'])
plt.title('Box Plot')
plt.xlabel('Total Bill vs Tip')
plt.ylabel('Money')
plt.show()

# Violin Plots
plt.violinplot(tips[['total_bill', 'tip']])
plt.xticks([1, 2], ['total_bill', 'tip'])
plt.title('Violin Plot')
plt.xlabel('Total Bill vs Tip')
plt.ylabel('Money')
plt.show()


















# %% 6. GridSearch key hyperparameters with RepeatedStratifiedKFold
# https://medium.com/@chaudhurysrijani/tuning-of-adaboost-with-computational-complexity-8727d01a9d20#:~:text=Explore%20the%20number%20of%20trees,often%20hundreds%2C%20if%20not%20thousands.

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

model = dtc(random_state=0)

# define the grid of values to search
grid = dict()
grid['max_depth'] = [2, 3, 4, 5, 6]
grid['min_samples_leaf'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=0)

# define the grid search procedure
metric = 'f1'
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           cv=cv, scoring=metric, return_train_score=True)

# execute the grid search
grid_search.fit(X, y)

# summarize all scores that were evaluated
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (± %f) with: %r" % (mean, stdev, param))

# Saving GridSearch result to dataframe
gsResult = pd.DataFrame(grid_search.cv_results_)

# summarize the best score and configuration
# print("Best test score: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
indexBest = grid_search.best_index_
testScoreStd = gsResult.at[indexBest,'std_test_score']
trainScoreMean = gsResult.at[indexBest,'mean_train_score']
trainScoreStd = gsResult.at[indexBest,'std_train_score']
print(f'Best test score: {grid_search.best_score_:.6f} (± {testScoreStd:.6f} using {grid_search.best_params_})')
print(f'Train score: {trainScoreMean:.6f} (± {trainScoreStd:.6f})')

# %% 7. 輸出 bestModel 決策樹

bestModel = grid_search.best_estimator_

# 繪製決策樹
plot_tree(bestModel, 
          feature_names = inputData.columns, 
          class_names = ['O', 'X'], 
          filled = True, 
          rounded = True)

# 另存圖檔
md = grid_search.best_params_['max_depth']
msl = grid_search.best_params_['min_samples_leaf']
figureName = f'{analDate} {analKey} {dataVersion} {analProblem} model {metric} (md {md}, msl {msl}).png'
plt.savefig(figureName)
