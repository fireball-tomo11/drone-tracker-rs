import pyrealsense2 as rs
import time
import cv2
import numpy as np

# 保存ファイル名
filename = "drone_test_flight.bag"

# 設定
pipeline = rs.pipeline()
config = rs.config()

# 解像度設定 (D435の推奨設定)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 録画設定 (ここが重要)
config.enable_record_to_file(filename)

print(f"録画を開始します: {filename}")
print("終了するには 'q' を押してください。")

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # プレビュー表示
        color_image = np.asanyarray(color_frame.get_data())
        cv2.putText(color_image, "REC", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Recording', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    print("録画終了。")