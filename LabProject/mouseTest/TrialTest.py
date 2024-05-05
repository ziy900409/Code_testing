import pygame
import sys
import math
import csv
import numpy as np
# %%
def edge_calculate(circle_x, circle_y, edge_cir_x, edge_cir_y):
    sin_angle = abs(circle_y - edge_cir_y) \
                    / np.sqrt((circle_x - edge_cir_x)**2 + (circle_y - edge_cir_y)**2)
    cos_angle = abs(circle_x - edge_cir_x) \
                    / np.sqrt((circle_x - edge_cir_x)**2 + (circle_y - edge_cir_y)**2)
                # 第一象限
    if circle_x - edge_cir_x > 0 and circle_y - edge_cir_y > 0:
        edge_point = [(circle_x + circle_radius*cos_angle),
                    (circle_y + circle_radius*sin_angle)]
    # 第二象限
    elif circle_x - edge_cir_x < 0 and circle_y - edge_cir_y > 0:
        edge_point = [(circle_x - circle_radius*cos_angle),
                    (circle_y + circle_radius*sin_angle)]
    # 第三象限
    elif circle_x - edge_cir_x < 0 and circle_y - edge_cir_y < 0:
        edge_point = [(circle_x - circle_radius*cos_angle),
                    (circle_y - circle_radius*sin_angle)]
    # 第四象限
    elif circle_x - edge_cir_x > 0 and circle_y - edge_cir_y < 0:
        edge_point = [(circle_x + circle_radius*cos_angle),
                    (circle_y - circle_radius*sin_angle)]
    # 如果躺在X軸上
    elif circle_y - edge_cir_y == 0:
    # 在周圍圓的右側
        if circle_x - edge_cir_x > 0:
            edge_point = [(circle_x + circle_radius),
                         (circle_y)]
        # 在周圍圓的左側
        elif circle_x - edge_cir_x < 0:
                 edge_point = [(circle_x - circle_radius),
                              (circle_y)]
    # 如果躺在Y軸上
    elif circle_x - edge_cir_x == 0:
        # 在周圍圓的上方
        if circle_y - edge_cir_y > 0:
            edge_point = [(circle_x),
                        (circle_y + circle_radius)]
        # 在周圍圓的下方
        elif circle_y - edge_cir_y < 0:
             edge_point = [(circle_x),
                             (circle_y - circle_radius)]
    return edge_point
# %% 基礎參數設定及初始化
# ------------基本受測資料------------------------
Amplitudes = [400, 800]
Widths = [40, 80]
Participant = "S01"
Condition = "S2"
Block = "01"
# -------------定义颜色--------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
CENTER_COLOR = (220, 190, 255)
PURPLE = (220, 190, 255)  # 淡紫色
# 定义圆的初始位置和大小
circle_radius = 20
# 记录已经拖拽到的位置数量
completed_positions = 0
record_edge = 1
# 标记是否有活动圆圈
active_circle = False
circle_touched = False
# 保存鼠标軌跡和按鍵事件的列表
mouse_click_events = []
mouse_positions = []
edge_circle = []

# 設定採樣間隔（毫秒）
sampling_interval = 10  # 每10毫秒採樣一次
# 計時器初始化
next_sample_time = pygame.time.get_ticks() + sampling_interval
# 設定不同難度
# 定义周围圆的数量
num_surrounding_circles = 14
# 設定多個不同難度的測試
tests = [
    {"surrounding_circle_radius": 40, 'target_amplitudes': 400}
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
    # 獲取滑鼠位置
    mouse_pos = pygame.mouse.get_pos()
    # 在視窗上顯示文字信息
    mouse_info = font.render(f"Mouse Position: ({mouse_pos[0]}, {mouse_pos[1]})", True, (0, 0, 0))
    window.blit(mouse_info, (20, 100))
    # 計算出所有周圍圓圈的所在位置
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
    # 繪製周圍圓的外框，根据是否被碰到选择绘制的颜色
    for (x, y) in surrounding_circles:
        if circle_touched:
            pygame.draw.circle(window, BLACK, (x, y),
                               surrounding_circle_radius, width=2)
            pygame.draw.circle(window, PURPLE, (surrounding_circles[completed_positions+1]),
                               surrounding_circle_radius)
        else:
            pygame.draw.circle(window, BLACK, (x, y), surrounding_circle_radius, width=2)
    # 绘制圆心标记数字
    number_text = font.render(str(completed_positions+1), True, RED)
    if completed_positions >= 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions+1])
        window.blit(number_text, text_rect)
    # 利用 record edge 讓中心圓位置只被紀錄一次
    if record_edge == completed_positions+1:
        record_edge += 1
        # 紀錄周圍圓的位置
        edge_circle.append((Participant, Condition, Block, # 受試者, 條件, 第幾次測試
                            (current_test_index + 1), (completed_positions+1), # sequemce, trial
                            current_test['target_amplitudes'], surrounding_circle_radius, # ampltude, width
                            "EdgeCirclePos", surrounding_circles[completed_positions+1], # event, edge circle position 
                            pygame.time.get_ticks())) # time
    
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
            """
            1. 判斷中心圓座標與周圍圓座標的相對位置，切分為四象限，以定義直角三角形
            2. 在中心圓周上找出離周圍圓圓心最遠的點，兩者相減必須小於周圍圓的半徑
            """                
            # 定義指定周圍圓的圓心
            edge_cir_x = surrounding_circles[completed_positions+1][0]
            edge_cir_y = surrounding_circles[completed_positions+1][1]
            edge_point = edge_calculate(circle_x, circle_y, edge_cir_x, edge_cir_y)
            # 計算周圍圓與中心圓的最遠位置的距離
            distance = math.sqrt((edge_point[0] - edge_cir_x) ** 2 + (edge_point[1] - edge_cir_y) ** 2)
            # 距離必須小於周圍圓的半徑
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
            if active_circle:
                # 检测是否拖拽到了周围的圆圈位置
                # 如果有拖曳到，就标记相应的外圆圈被碰到
                for i, (x, y) in enumerate(surrounding_circles):
                    # 定義指定周圍圓的圓心
                    edge_cir_x = surrounding_circles[completed_positions+1][0]
                    edge_cir_y = surrounding_circles[completed_positions+1][1]
                    edge_point = edge_calculate(circle_x, circle_y, edge_cir_x, edge_cir_y)
                    # 計算周圍圓與中心圓的最遠位置的距離
                    distance = math.sqrt((edge_point[0] - edge_cir_x) ** 2 + (edge_point[1] - edge_cir_y) ** 2)
                    if distance <= surrounding_circle_radius:
                       circle_touched = True
                    else:
                        circle_touched = False
    
    # 固定時間採樣一次滑鼠位置，檢查是否到達採樣時間
    if pygame.time.get_ticks() >= next_sample_time:
        # 保存滑鼠位置
        mouse_pos = pygame.mouse.get_pos()
        # 更新下一次採樣時間
        next_sample_time = pygame.time.get_ticks() + sampling_interval
        # 保存鼠标移动轨迹
        mouse_positions.append((Participant, Condition, Block, 
                                (current_test_index + 1), (completed_positions+1), # sequemce, trial
                                  current_test['target_amplitudes'], surrounding_circle_radius,
                                  mouse_pos, current_time))

    # 如果完成了所有位置，切換到下一個測試
    if completed_positions+1 == num_surrounding_circles:
        # 重置完成的位置
        completed_positions = 0
        record_edge = 1
        circle_touched = False
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
                # for part, cond, blo, sec, trial, amp, wid, pos, time in mouse_trajectory:
                #     writer.writerow([part, cond, blo, sec, trial, amp, wid, 'MOUSEMOTION', time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, pos, time in mouse_positions:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, 'MOUSEPOS', time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, event, pos, time in mouse_click_events:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, event, time, pos[0], pos[1]])
                for part, cond, blo, sec, trial, amp, wid, event, pos, time in edge_circle:
                    writer.writerow([part, cond, blo, sec, trial, amp, wid, event, time, pos[0], pos[1]])
            # 如果已經達到最後一個測試，則退出遊戲
            pygame.quit()
            sys.exit()

    # 如果有活动圆圈，根据鼠标位置更新圆圈位置
    if active_circle:
        circle_x, circle_y = pygame.mouse.get_pos()

    pygame.display.update()
