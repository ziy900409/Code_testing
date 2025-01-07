"""
1. 偵測小球
2. 偵測準心？

1. 偵測影片中小球的"位置"、"出現時間"、"消失時間"
2. 偵測各球體之間的距離
3. 是否可以偵測準心移動軌跡？or 另外寫程式同步，使用某個按鍵做trigger，記錄滑鼠的移動情形


"""

# %%
import cv2
import numpy as np
import csv
import ffmpeg
# 設定影片路徑 (.mkv 檔案)
video_path = "your_video.mkv"  # 替換為你的 .mkv 檔案路徑

folder = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\5. VideoRecord\2024 Shanghai Major\non-sym\S3\Spider\A\\"
video_path = folder + "2024-11-26 14-47-35.mkv"  # 替換為你的影片路徑

# %%
import cv2
import numpy as np
import csv

def detect_ball_position(frame, dp=1.2, min_dist=50, param1=50, param2=30, min_radius=10, max_radius=50):
    """
    偵測影片幀中的球體位置 (基於霍夫圓變換)
    :param frame: 當前幀
    :param dp: 霍夫圓變換的累積分辨率參數
    :param min_dist: 圓心之間的最小距離
    :param param1: Canny 邊緣檢測的高閾值
    :param param2: 霍夫圓變換的圓檢測閾值
    :param min_radius: 檢測圓的最小半徑
    :param max_radius: 檢測圓的最大半徑
    :return: 檢測到的圓列表，每個圓為 (x, y, radius)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉為灰階
    gray = cv2.medianBlur(gray, 5)  # 中值濾波，平滑影像
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))  # 將座標四捨五入為整數
        return circles[0, :]  # 返回檢測到的圓形清單
    return []

def save_data_to_csv(data, filename="ball_positions.csv"):
    """
    保存球體位置數據到 CSV 文件
    :param data: 包含 (秒數, X, Y, 半徑) 的數據列表
    :param filename: 輸出 CSV 文件名
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "X", "Y", "Radius"])
        writer.writerows(data)

cap = cv2.VideoCapture(video_path)

# 解析影片資訊
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # 設置默認 FPS
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 儲存球體數據
ball_positions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("影片讀取完成。")
        break

    # 當前幀對應的秒數
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    time_sec = frame_idx / fps

    # 偵測球體位置
    circles = detect_ball_position(frame)
    for circle in circles:
        x, y, radius = circle
        ball_positions.append([time_sec, x, y, radius])

        # 在畫面上繪製球體
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {time_sec:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 顯示影片
    cv2.imshow("Video", frame)

    # 按 Q 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 保存數據到 CSV
save_data_to_csv(ball_positions)

# 釋放資源
cap.release()
cv2.destroyAllWindows()

print(f"檢測完成，球體位置已保存到 'ball_positions.csv'")

# %%

import cv2
import numpy as np
import os

# 載入圖片
image_path = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\5. VideoRecord\2024 Shanghai Major\non-sym\S3\Spider\A\2025-01-07 094848.png"
# 確認圖片是否存在
if not os.path.exists(image_path):
    print("圖片路徑不存在，請檢查路徑是否正確！")
else:
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("圖片讀取失敗，請檢查檔案格式是否正確！")
    else:
        # 將圖片轉換為 HSV 色彩空間
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        print("圖片成功讀取並轉換為 HSV！")


# image = cv2.imread(image_path)

# 將圖片轉換為 HSV 色彩空間
# image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義青藍色球體的 HSV 範圍
lower_blue = np.array([90, 150, 150])  # 根據球體顏色調整
upper_blue = np.array([120, 255, 255])

# 創建遮罩
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 屏蔽上方的遊戲計時器區域
mask[0:100, :] = 0  # 遮擋上方 0 到 100 像素的區域

# 形態學操作去除雜訊
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 尋找輪廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 檢測球體
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # 過濾小物件
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # 篩選半徑和位置
        if radius > 5:  # 半徑篩選 + 下方位置限制
            # 繪製圓形和中心點
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # 綠色圓圈
            cv2.circle(image, center, 5, (0, 0, 255), -1)  # 紅色中心點

# 顯示結果
cv2.imshow("Filtered Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%

import cv2
import numpy as np

def process_frame(frame, lower_color, upper_color, y_limit=100, max_radius=50, min_radius=5, min_area=100):
    """
    處理每幀以檢測球體
    :param frame: 當前幀
    :param lower_color: HSV 顏色範圍的下界
    :param upper_color: HSV 顏色範圍的上界
    :param y_limit: 最小 y 限制（例如屏蔽畫面上方區域）
    :param min_radius: 最小檢測半徑
    :param min_area: 最小檢測面積
    :return: 標記後的幀
 
    處理每幀以檢測球體，並在屏蔽區域添加透明色塊
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 創建遮罩
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 屏蔽畫面上方區域
    mask[0:y_limit, :] = 0

    # 可視化屏蔽區域（添加透明色塊）
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], y_limit), (0, 0, 255), -1)  # 紅色矩形
    alpha = 0.3  # 透明度
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 形態學操作去除雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 尋找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # 篩選半徑和位置
            if radius > min_radius and y > y_limit:
                # 繪製圓形和中心點
                cv2.circle(frame, center, radius, (0, 255, 0), 2)  # 綠色圓圈
                cv2.circle(frame, center, 5, (0, 0, 255), -1)  # 紅色中心點

    return frame

def detect_ball_in_video(video_path, lower_color, upper_color, output_path=None):
    """
    在影片中進行球體偵測
    :param video_path: 影片路徑
    :param lower_color: HSV 顏色範圍的下界
    :param upper_color: HSV 顏色範圍的上界
    :param output_path: 如果需要保存輸出的影片，提供保存路徑
    """
    cap = cv2.VideoCapture(video_path)

    # 檢查影片是否打開成功
    if not cap.isOpened():
        print("無法打開影片，請確認路徑是否正確。")
        return

    # 影片資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 如果需要保存輸出影片
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("影片處理完成。")
            break

        # 處理當前幀
        processed_frame = process_frame(frame, lower_color, upper_color)

        # 顯示結果
        cv2.imshow("Ball Detection", processed_frame)

        # 保存影片q
        if output_path:
            out.write(processed_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# 定義 HSV 顏色範圍（根據球體顏色調整）
lower_blue = np.array([90, 150, 150])  # 根據球體顏色調整
upper_blue = np.array([120, 255, 255])

# 影片路徑
# video_path = r"D:\BenQ_Project\git\Code_testing\LabProject\ComputerVision\video.mp4"  # 替換為你的影片路徑

# 保存影片的輸出路徑（選擇性）
output_path = folder + "output_video.avi"

# 執行球體偵測
detect_ball_in_video(video_path, lower_blue, upper_blue, output_path)

