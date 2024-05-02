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

# %%
raw_data = pd.read_csv(r"D:\BenQ_Project\FittsDragDropTest\mouse_data.csv")

# %%
"""
找出所有 trail by trail 的順序及路徑
"""























