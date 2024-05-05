import pygame
import sys
import math

# 初始化 Pygame
pygame.init()

# 设置窗口大小
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Drag and Drop to Finish Test')

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)  # 淡紫色

# 定义圆的初始位置和大小
circle_radius = 20
circle_x = WINDOW_WIDTH // 2
circle_y = WINDOW_HEIGHT // 2

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
    x = circle_x + int(circle_radius * 10 * math.cos(angle_rad))
    y = circle_y + int(circle_radius * 10 * math.sin(angle_rad))
    surrounding_circles.append((x, y))

# 在循环外定义一个列表来跟踪每个外圆圈是否被碰到
circle_touched = [False] * num_surrounding_circles

# 记录已经拖拽到的位置数量
completed_positions = 0

# 标记是否有活动圆圈
active_circle = False

# 游戏循环
while True:
    window.fill(WHITE)

    # 绘制周围圆的外框，根据是否被碰到选择绘制的颜色
    for i, (x, y) in enumerate(surrounding_circles):
        if circle_touched[i]:
            pygame.draw.circle(window, PURPLE, (x, y), surrounding_circle_radius)
        else:
            pygame.draw.circle(window, BLACK, (x, y), surrounding_circle_radius, width=2)
        # 绘制数字到周围圆
        font = pygame.font.Font(None, 24)
        number_text = font.render(number_list[i], True, RED)
        text_rect = number_text.get_rect(center=(x, y))
        window.blit(number_text, text_rect)
    
    # 获取鼠标位置
    mouse_x, mouse_y = pygame.mouse.get_pos()
    # 在左下角绘制鼠标位置
    mouse_position_text = font.render(f"Mouse Position: ({mouse_x}, {mouse_y})", True, RED)
    window.blit(mouse_position_text, (10, WINDOW_HEIGHT - 30))  # 放在窗口左下角

    # 绘制中心圆
    pygame.draw.circle(window, BLACK, (circle_x, circle_y), circle_radius)

    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 检测鼠标是否与圆圈相交
            distance = math.sqrt((circle_x - event.pos[0])**2 + (circle_y - event.pos[1])**2)
            if distance <= circle_radius:
                active_circle = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if active_circle:
                active_circle = False
                # 检测是否拖拽到了周围的圆圈位置
                # 如果有拖曳到，就 completed_positions+1，并标记相应的外圆圈被碰到
                for i, (x, y) in enumerate(surrounding_circles):
                    distance = math.sqrt((x - circle_x)**2 + (y - circle_y)**2)
                    if distance <= surrounding_circle_radius:
                        completed_positions += 1
                        circle_touched[i] = True
                        break
        elif event.type == pygame.MOUSEMOTION:
            if active_circle:
                # 检测是否拖拽到了周围的圆圈位置
                # 如果有拖曳到，就标记相应的外圆圈被碰到
                for i, (x, y) in enumerate(surrounding_circles):
                    distance = math.sqrt((x - mouse_x)**2 + (y - mouse_y)**2)
                    if distance <= surrounding_circle_radius:
                        circle_touched[i] = True
                    else:
                        circle_touched[i] = False

    # 如果拖拽到了所有位置，结束游戏
    if completed_positions == num_surrounding_circles:
        pygame.quit()
        sys.exit()

    # 如果有活动圆圈，根据鼠标位置更新圆圈位置
    if active_circle:
        circle_x, circle_y = pygame.mouse.get_pos()

    pygame.display.update()
