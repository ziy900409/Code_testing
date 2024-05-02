import pygame
import sys
import math
import random
import numpy as np
# %% 參數定義
# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
Amplitudes = [400, 800]
Widths = [40, 80]
# 定义周围圆的数量
num_surrounding_circles = 14
# 設定多個不同難度的測試
tests = [
    {"surrounding_circle_radius": 40, 'target_amplitudes': 400},
    {"surrounding_circle_radius": 20, 'target_amplitudes': 800}
]

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
     

# %%
# 初始化 Pygame
pygame.init()

# 设置窗口大小和标题
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Drag and Drop to Finish Test')



# 定义圆的初始位置和大小
circle_radius = 20
# 定義中心點位置
center_x = WINDOW_WIDTH // 2
center_y = WINDOW_HEIGHT // 2
# 計算起始圓心位置

for i in range(len(all_combinations)):
    all_combinations[i]['circle_x'] = center_x + all_combinations[i]['target_amplitudes']
    all_combinations[i]['circle_y'] = center_y + all_combinations[i]['target_amplitudes']

# 定义周围圆的数量和半径
num_surrounding_circles = 15

# 计算中心圆的对角线方向
diagonal_angle = math.radians(45)

# 生成填入数字的列表
number_list = [str(i+1) for i in range(num_surrounding_circles)]

# 定义圆的初始位置和大小
circle_x = center_x
circle_y = center_y
# 记录已经拖拽到的位置数量
completed_positions = 0

# 标记是否有活动圆圈
active_circle = False

# 保存鼠标軌跡和按鍵事件的列表
mouse_trajectory = []
mouse_click_events = []



# 當前測試索引
current_test_index = 0

# 選擇第一個測試
current_test = tests[current_test_index]

# 游戲循环
while True:
    window.fill(WHITE)

    # 根據當前測試參數繪製周圍的圓
    # num_surrounding_circles = current_test["num_surrounding_circles"]
    surrounding_circle_radius = current_test["surrounding_circle_radius"]
    angle_step = math.radians(360 / num_surrounding_circles)
    surrounding_circles = []
    for i in range(num_surrounding_circles):
        if i % 2 == 0:
            angle_rad = int(i / 2) * angle_step
        else:
            angle_rad = int(i / 2) * angle_step + 3.14
        x = center_x + int(circle_radius * 10 * math.cos(angle_rad))
        y = center_y + int(circle_radius * 10 * math.sin(angle_rad))
        surrounding_circles.append((x, y))
    circle_x = surrounding_circles[0][0]
    circle_y = surrounding_circles[0][1]   

    # 繪製周圍圓的外框
    for (x, y) in surrounding_circles:
        pygame.draw.circle(window, BLACK, (x, y), surrounding_circle_radius, width=2)
    # 绘制圆心标记数字
    font = pygame.font.Font(None, 24)
    number_text = font.render(str(completed_positions+1), True, RED)
    if completed_positions > 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions])
        window.blit(number_text, text_rect)
    elif completed_positions == 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions])
        window.blit(number_text, text_rect)
    # 繪製中心圓
    pygame.draw.circle(window, BLACK, (circle_x, circle_y), circle_radius)

    # 處理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 檢測鼠標是否與中心圓相交
            distance = math.sqrt((circle_x - event.pos[0]) ** 2 + (circle_y - event.pos[1]) ** 2)
            if distance <= circle_radius:
                active_circle = True
                mouse_click_events.append(("MOUSEBUTTONDOWN", pygame.time.get_ticks()))
        elif event.type == pygame.MOUSEBUTTONUP:
            if active_circle:
                active_circle = False
                # 檢測是否拖拽到了周圍的圓圈位置
                # 如果拖拽到，就 completed_positions+1
                for i, (x, y) in enumerate(surrounding_circles):
                    distance = math.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2)
                    if distance <= surrounding_circle_radius:
                        completed_positions += 1
                        mouse_click_events.append(("MOUSEBUTTONUP", pygame.time.get_ticks()))
                        break

    # 如果完成了所有位置，切換到下一個測試
    if completed_positions == num_surrounding_circles:
        # 重置完成的位置
        completed_positions = 0
        # 切換到下一個測試
        current_test_index += 1
        if current_test_index < len(tests):
            current_test = tests[current_test_index]
        else:
            # 如果已經達到最後一個測試，則退出遊戲
            pygame.quit()
            sys.exit()

    # 如果有活動圓圈，根據鼠標位置更新圓圈位置
    if active_circle:
        circle_x, circle_y = pygame.mouse.get_pos()

    pygame.display.update()
