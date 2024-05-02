
"""
待處理功能：

3. 設定不同難度問題
    3.1. 不同難度應該隨機出現

已解決
1. 拖曳順序問題 
    1.1. 完成各個trial後，出現下一個數字，數字應該出現在對角線
    1.2. trial 完成的判定為放開左鍵
2. 輸出檔案問題
    2.1. 游標位置紀錄應從第一次正確點擊開始，並且至最後一個trial完成後結束

程式架構
1. 產生圓圈： 
    1.1. 建立一個圓圈類，每個圓圈物件代表一個拖曳任務目標。
    1.2. 在每個拖曳任務開始前，產生一個或多個這樣的圓圈物件並顯示在螢幕上。
2. 處理滑鼠拖曳事件： 
    2.1. 使用 pygame.MOUSEBUTTONDOWN 和 pygame.MOUSEMOTION 事件來處理滑鼠的拖曳操作。
    2.2. 當滑鼠按下並移動時，更新拖曳物件的位置。
    2.3. 當滑鼠按鍵釋放，當成一次事件結束，並重新更新圓圈位置
3. 偵測拖曳碰撞： 
    3.1. 在每次滑鼠移動時，偵測滑鼠是否與任何拖曳物件相交，
    3.2. 如果相交，則將目前拖曳物件標記為活動物件 (改成相交且滑鼠左鍵or中鍵按壓)。
    3.3. 當滑鼠釋放時，偵測是否有活動對象，並根據需要執行相應的操作（如移動、放置等）。
4. 計算操作時間與距離： 
    4.1. 在每次拖曳操作完成後，記錄拖曳操作的起始位置、結束位置以及所花費的時間。
    4.2. 根據 Fitts's Law 的公式，計算操作的 ID（Index of Difficulty）和各項指標（如 MT、TP）。
5. 顯示結果： 
    5.1. 將每次拖曳操作的結果（如操作時間、ID、TP 等）顯示在螢幕上，以便使用者查看。

@author: Hsin.YH.Yang, written by May 02 2024
"""
# %% import library
import pygame
import sys
import math
import random
import csv
# %% 參數定義
# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
Amplitudes = [400, 800]
Widths = [40, 80]
Participant = "S01"
Condition = "S2"
Block = "01"
# 設定不同難度
# 定义周围圆的数量
num_surrounding_circles = 14
# 設定多個不同難度的測試
tests = [
    {"surrounding_circle_radius": 40, 'target_amplitudes': 400},
    {"surrounding_circle_radius": 20, 'target_amplitudes': 250}
]
#{"surrounding_circle_radius": 20, 'target_amplitudes': 250}
# 保存所有可能的組合
all_combinations = []

# 外部循環迭代surrounding_circle_radius
for radius_info in tests:
    radius = radius_info['surrounding_circle_radius']
    # 內部循環迭代target_amplitudes
    for amplitude_info in tests:
        amplitude = amplitude_info['target_amplitudes']
        # 將此組合添加到所有組合列表中
        all_combinations.append({"surrounding_circle_radius": radius, "target_amplitudes": amplitude})
     
# 當前測試索引
current_test_index = 0

# 選擇第一個測試
current_test = all_combinations[current_test_index]
# %%
# 初始化 Pygame
pygame.init()

# 设置窗口大小和标题
infoObject = pygame.display.Info()
WINDOW_WIDTH = infoObject.current_w-100
WINDOW_HEIGHT = infoObject.current_h-100
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Drag and Drop to Finish Test')



# 定义圆的初始位置和大小
circle_radius = 20
# 定義中心點位置
center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2

# 計算起始圓心位置

for i in range(len(all_combinations)):
    all_combinations[i]['circle_x'] = center_x + \
          all_combinations[i]['target_amplitudes'] * math.cos(0)
    all_combinations[i]['circle_y'] = center_y + \
        all_combinations[i]['target_amplitudes'] * math.sin(0)


circle_x = all_combinations[0]['circle_x'] 
circle_y = all_combinations[0]['circle_y'] 
# 计算中心圆的对角线方向
diagonal_angle = math.radians(45)

# 生成填入数字的列表
number_list = [str(i+1) for i in range(num_surrounding_circles)]


# 记录已经拖拽到的位置数量
completed_positions = 0

# 标记是否有活动圆圈
active_circle = False

# 保存鼠标軌跡和按鍵事件的列表
mouse_trajectory = []
mouse_click_events = []


# 游戏循环
while True:
    window.fill(WHITE)
    # 根據當前測試參數繪製周圍的圓
    surrounding_circle_radius = current_test["surrounding_circle_radius"]
    # 定義字體大小
    font = pygame.font.Font(None, 36)
    # 顯示 task 資訊
    # 顯示第幾個任務
    tests_info = font.render(f"Sequence {current_test_index + 1} of {len(all_combinations)}",
                              True, (0, 0, 0))
    tests_para = font.render(f"(A = {current_test['target_amplitudes']}, W = {surrounding_circle_radius})",
                               True, (0, 0, 0))
    window.blit(tests_info, (20, 20))
    window.blit(tests_para, (20, 60))

    angle_step = math.radians(360 / num_surrounding_circles)
    surrounding_circles = []
    for i in range(num_surrounding_circles):
        if i % 2 == 0:
            angle_rad = int(i / 2) * angle_step
        else:
            angle_rad = int(i / 2) * angle_step + 3.14
        x = center_x + int(current_test['target_amplitudes'] * math.cos(angle_rad))
        y = center_y + int(current_test['target_amplitudes'] * math.sin(angle_rad))
        surrounding_circles.append((x, y))
    
    # 繪製周圍圓的外框
    for (x, y) in surrounding_circles:
        pygame.draw.circle(window, BLACK, (x, y), surrounding_circle_radius, width=2)
    # 绘制圆心标记数字
    
    number_text = font.render(str(completed_positions+1), True, RED)
    if completed_positions > 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions+1])
        window.blit(number_text, text_rect)
    elif completed_positions == 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions+1])
        window.blit(number_text, text_rect)
    # 繪製中心圓
    pygame.draw.circle(window, BLACK, (circle_x, circle_y), surrounding_circle_radius*0.6)

    # 設置採樣間隔（毫秒）
    # 處理事件
    for event in pygame.event.get():
        current_time = pygame.time.get_ticks()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 檢測鼠標是否與中心圓相交
            distance = math.sqrt((circle_x - event.pos[0]) ** 2 + (circle_y - event.pos[1]) ** 2)
            if distance <= circle_radius:
                active_circle = True
                mouse_click_events.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                                           (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                           current_test['target_amplitudes'], surrounding_circle_radius, # ampltude, width
                                           "MOUSEBUTTONDOWN", event.pos, pygame.time.get_ticks())) # event, mouse position, time
        elif event.type == pygame.MOUSEBUTTONUP:
            if active_circle:
                active_circle = False
                # 檢測是否拖拽到了周圍的圓圈位置
                # 如果拖拽到，就 completed_positions+1
                find = 0
                distance = math.sqrt((surrounding_circles[completed_positions+1][0] - circle_x) ** 2 + \
                                     (surrounding_circles[completed_positions+1][1] - circle_y) ** 2)
                # 改成只有進到有標記數字的圓圈才算成功
                if distance <= surrounding_circle_radius:
                    completed_positions += 1
                    mouse_click_events.append((Participant, Condition, Block,
                                               (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                               current_test['target_amplitudes'], surrounding_circle_radius,
                                               "MOUSEBUTTONUP_SUCC", event.pos, pygame.time.get_ticks()))
                    find = 1
                    break
                # 如果找不到有在任何圓圈內，但是又偵測到MOUSEBUTTONUP，算失敗
                if find == 0:
                    completed_positions += 1
                    mouse_click_events.append((Participant, Condition, Block, 
                                               (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                               current_test['target_amplitudes'], surrounding_circle_radius,
                                               "MOUSEBUTTONUP_FAIL", event.pos, pygame.time.get_ticks()))
                    break
                    
        elif event.type == pygame.MOUSEMOTION:
            # 保存鼠标移动轨迹
            mouse_trajectory.append((Participant, Condition, Block, 
                                     (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                     current_test['target_amplitudes'], surrounding_circle_radius,
                                     event.pos, current_time))
            # 更新下一次採樣時間
            print((current_test['target_amplitudes'], surrounding_circle_radius,
                   event.pos, current_time))
    # 獲取滑鼠位置
    mouse_pos = pygame.mouse.get_pos()
    # 在視窗上顯示文字信息
    mouse_info = font.render(f"Mouse Position: ({mouse_pos[0]}, {mouse_pos[1]})", True, (0, 0, 0))
    window.blit(mouse_info, (20, 100))
    # 如果完成了所有位置，切換到下一個測試
    if completed_positions+1 == num_surrounding_circles:
        # 重置完成的位置
        completed_positions = 0
        # 切換到下一個測試
        current_test_index += 1
        if current_test_index < len(all_combinations):
            current_test = all_combinations[current_test_index]
            circle_x = all_combinations[current_test_index]['circle_x']
            circle_y = all_combinations[current_test_index]['circle_y']
        else:
            # 將滑鼠軌跡和事件資料寫入 CSV 檔案
            with open(r'D:\BenQ_Project\FittsDragDropTest\mouse_data.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Participant', 'Condition', 'Block',
                                  'Sequence', 'Trial',
                                  'Amplitudes', 'Width',
                                'Event', 'time', 'Pos_x', 'Pos_y'])
                for part, cond, blo, sec, trial, amp, wid, pos, time in mouse_trajectory:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, 'MOUSEMOTION', time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, event, pos, time in mouse_click_events:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, event, time, pos[0], pos[1]])
            # 如果已經達到最後一個測試，則退出遊戲
            pygame.quit()
            sys.exit()

    # 如果有活动圆圈，根据鼠标位置更新圆圈位置
    if active_circle:
        circle_x, circle_y = pygame.mouse.get_pos()

    pygame.display.update()

