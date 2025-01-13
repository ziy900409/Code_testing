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

folder = r"D:\BenQ_Project\01_UR_lab\2024_11 Shanghai CS Major\5. VideoRecord\2024 Shanghai Major\non-sym\S9\Spider\C\\"
video_path = folder + "2024-12-02 16-19-25.mkv"  # 替換為你的影片路徑

# %%

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
lower_blue = np.array([203, 206, 60])  # 根據球體顏色調整
upper_blue = np.array([116, 129, 54])

# 影片路徑
# video_path = r"D:\BenQ_Project\git\Code_testing\LabProject\ComputerVision\video.mp4"  # 替換為你的影片路徑

# 保存影片的輸出路徑（選擇性）
output_path = folder + "output_video.avi"

# 執行球體偵測
detect_ball_in_video(video_path, lower_blue, upper_blue, output_path)

