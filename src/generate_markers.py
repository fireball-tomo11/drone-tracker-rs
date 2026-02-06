import cv2
import numpy as np
import os

# ... (以下保存処理) ...
current_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(current_dir, "..", "markers")
os.makedirs(save_dir, exist_ok=True)

# ArUco辞書の設定 (4x4, 50個)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 生成するIDリスト
ids = [0, 1, 2, 3, 4, 10]

print(f"'{save_dir}' フォルダにマーカーを生成します...")

for marker_id in ids:
    # 200x200ピクセルのマーカー画像を生成
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    
    # 枠線をつけて切りやすくする (オプション)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    filename = f"{save_dir}/marker_id_{marker_id}.png"
    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")

print("完了。印刷して床とドローンに配置してください。")