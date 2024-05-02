import pygame
import sys

# 初始化Pygame
pygame.init()

# 設置窗口大小
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Input Blocks"
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(WINDOW_TITLE)

# 定義顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 創建文本輸入框
font = pygame.font.Font(None, 32)

# 定義三個輸入框的矩形區域
input_rect1 = pygame.Rect(300, 200, 200, 32)
input_rect2 = pygame.Rect(300, 300, 200, 32)
input_rect3 = pygame.Rect(300, 400, 200, 32)

# 初始化三個輸入框的文字
input_text1 = ''
input_text2 = ''
input_text3 = ''

# 主循環
while True:
    # 處理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            # 按下 Enter 鍵時，將三個輸入框中的文字輸出到控制台
            if event.key == pygame.K_RETURN:
                print("參數1:", input_text1)
                print("參數2:", input_text2)
                print("參數3:", input_text3)
                # 清空輸入框中的文字
                input_text1 = ''
                input_text2 = ''
                input_text3 = ''
            # 刪除最後一個字符
            elif event.key == pygame.K_BACKSPACE:
                if input_rect1.collidepoint(pygame.mouse.get_pos()):
                    input_text1 = input_text1[:-1]
                elif input_rect2.collidepoint(pygame.mouse.get_pos()):
                    input_text2 = input_text2[:-1]
                elif input_rect3.collidepoint(pygame.mouse.get_pos()):
                    input_text3 = input_text3[:-1]
            # 將鍵入的字符添加到輸入框中
            else:
                if input_rect1.collidepoint(pygame.mouse.get_pos()):
                    input_text1 += event.unicode
                elif input_rect2.collidepoint(pygame.mouse.get_pos()):
                    input_text2 += event.unicode
                elif input_rect3.collidepoint(pygame.mouse.get_pos()):
                    input_text3 += event.unicode

    # 渲染三個文本輸入框
    window.fill(WHITE)
    pygame.draw.rect(window, BLACK, input_rect1, 2)
    text_surface1 = font.render(input_text1, True, BLACK)
    window.blit(text_surface1, (input_rect1.x + 5, input_rect1.y + 5))

    pygame.draw.rect(window, BLACK, input_rect2, 2)
    text_surface2 = font.render(input_text2, True, BLACK)
    window.blit(text_surface2, (input_rect2.x + 5, input_rect2.y + 5))

    pygame.draw.rect(window, BLACK, input_rect3, 2)
    text_surface3 = font.render(input_text3, True, BLACK)
    window.blit(text_surface3, (input_rect3.x + 5, input_rect3.y + 5))

    pygame.display.flip()

    # 控制幀率
    pygame.time.Clock().tick(60)
