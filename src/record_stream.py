import pyrealsense2 as rs
import time
import cv2
import numpy as np
import os

# --- 保存先設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "..", "data")
os.makedirs(save_dir, exist_ok=True)

# ファイル名
file_name_only = "drone_test_ir.bag" # わかりやすく名前を変えました
full_save_path = os.path.join(save_dir, file_name_only)

# --- RealSense設定 ---
pipeline = rs.pipeline()
config = rs.config()

# 【変更点】カラーではなく、赤外線(Infrared 1)と深度(Depth)を有効化
# 解像度は848x480にすると画角をフルに使えます(D435の場合)
config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

# 録画設定
config.enable_record_to_file(full_save_path)

print(f"赤外線モードで録画を開始します: {full_save_path}")
print("終了するには 'q' を押してください。")

profile = pipeline.start(config)

# 【重要】ドットパターン投光器(Emitter)をOFFにする
# これがONだと画面中がドットだらけになり、マーカー認識に失敗します
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0.0) # 0 = OFF

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # 赤外線フレームを取得
        ir_frame = frames.get_infrared_frame(1)
        if not ir_frame:
            continue

        # そのまま白黒画像として取得
        ir_image = np.asanyarray(ir_frame.get_data())

        # 録画中表示
        cv2.putText(ir_image, "REC (IR Mode)", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
        cv2.imshow('Recording', ir_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("録画終了。")