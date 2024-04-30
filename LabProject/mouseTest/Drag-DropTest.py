
"""
待處理功能：


2. 輸出檔案問題
    2.1. 游標位置紀錄應從第一次正確點擊開始，並且至最後一個trial完成後結束
3. 設定不同難度問題

已解決
1. 拖曳順序問題 
    1.1. 完成各個trial後，出現下一個數字，數字應該出現在對角線
    1.2. trial 完成的判定為放開左鍵

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
"""
# %% import library
import pygame
import sys
import math
import random
# %% 參數定義
# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
Amplitudes = [400, 800]
Widths = [40, 80]
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

# 定义周围圆的数量和半径
num_surrounding_circles = 14
surrounding_circle_radius = 30

# 计算中心圆的对角线方向
diagonal_angle = math.radians(45)

# 生成填入数字的列表
number_list = [str(i+1) for i in range(num_surrounding_circles)]

# 计算周围圆的位置
angle_step = math.radians(360 / num_surrounding_circles)
surrounding_circles = []
for i in range(num_surrounding_circles):
    if i % 2 == 0: 
        angle_rad = int(i/2) * angle_step
    else:
        angle_rad = int(i/2) * angle_step + 3.14
    x = center_x + int(circle_radius * 10 * math.cos(angle_rad))
    y = center_y + int(circle_radius * 10 * math.sin(angle_rad))
    surrounding_circles.append((x, y))
# 定义圆的初始位置和大小
circle_x = surrounding_circles[0][0]
circle_y = surrounding_circles[0][1]
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

    # 绘制周围圆的外框
    for (x, y) in surrounding_circles:
        pygame.draw.circle(window, BLACK, (x, y), surrounding_circle_radius, width=2)
    
    # 绘制中心圆
    pygame.draw.circle(window, BLACK, (circle_x, circle_y), circle_radius)

    # 绘制圆心标记数字
    font = pygame.font.Font(None, 24)
    number_text = font.render(str(completed_positions+1), True, RED)
    if completed_positions > 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions+1])
        window.blit(number_text, text_rect)
    elif completed_positions == 0:
        text_rect = number_text.get_rect(center=surrounding_circles[completed_positions+1])
        window.blit(number_text, text_rect)

    # 設置採樣間隔（毫秒）
    sampling_interval = 100  # 例如，每100毫秒採樣一次
    # 定義下一次採樣時間
    next_sample_time = pygame.time.get_ticks() + sampling_interval

    # 处理事件
    for event in pygame.event.get():
        current_time = pygame.time.get_ticks()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 检测鼠标是否与中心圆相交
            distance = math.sqrt((circle_x  - event.pos[0])**2 + \
                                  (circle_y - event.pos[1])**2)
            if distance <= circle_radius:
                active_circle = True
                # 保存按鍵事件時間
                mouse_click_events.append(("MOUSEBUTTONDOWN", current_time))
        elif event.type == pygame.MOUSEBUTTONUP:
            if active_circle:
                active_circle = False
                
                # 检测是否拖拽到了周围的圆圈位置
                # 如果拖拽到，就 completed_positions+1
                for i, (x, y) in enumerate(surrounding_circles):
                    distance = math.sqrt((x - circle_x)**2 + (y - circle_y)**2)
                    if distance <= surrounding_circle_radius:
                        completed_positions += 1
                        # 保存按鍵事件時間
                        mouse_click_events.append(("MOUSEBUTTONUP", current_time))
                        break
        elif event.type == pygame.MOUSEMOTION and current_time >= next_sample_time:
            # 保存鼠标移动轨迹
            mouse_trajectory.append((event.pos, current_time))
            # 更新下一次採樣時間
            next_sample_time = current_time + sampling_interval
            print((event.pos, current_time))

    # 如果拖拽到了所有位置，结束游戏
    if completed_positions == (num_surrounding_circles + 1):
        pygame.quit()
        sys.exit()

    # 如果有活动圆圈，根据鼠标位置更新圆圈位置
    if active_circle:
        circle_x, circle_y = pygame.mouse.get_pos()

    pygame.display.update()

