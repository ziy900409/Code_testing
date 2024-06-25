import pygame
import sys

# 初始化 Pygame
pygame.init()

# 設置窗口大小
window_size = (800, 600)
window = pygame.display.set_mode(window_size)

# 顏色設置
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

# 圓圈參數
circle_x, circle_y = 400, 300
circle_radius = 50
circle_alpha = 128  # 透明度（0-255）

# 創建一個臨時表面
circle_surface = pygame.Surface((circle_radius * 2, circle_radius * 2), pygame.SRCALPHA)
circle_surface = circle_surface.convert_alpha()

# 填充臨時表面為透明
circle_surface.fill((0, 0, 0, 0))

# 在臨時表面上繪製圓圈
pygame.draw.circle(circle_surface, (BLACK[0], BLACK[1], BLACK[2], circle_alpha), (circle_radius, circle_radius), circle_radius)

# 主循環
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 填充窗口背景
    window.fill(BACKGROUND_COLOR)

    # 將臨時表面上的圓圈繪製到主屏幕上
    window.blit(circle_surface, (circle_x - circle_radius, circle_y - circle_radius))

    # 更新顯示
    pygame.display.flip()

# 退出 Pygame
pygame.quit()
sys.exit()
