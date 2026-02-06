import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import csv
import os
import datetime

# ==========================================
# ユーザー設定エリア
# ==========================================
MARKER_SIZE_REF = 0.15   # 床マーカーの実測値 [m]
MARKER_SIZE_DRONE = 0.1 # ドローンマーカーの実測値 [m]

# 基準マーカー座標 (ID: X, Y, Z)
REF_MARKERS_MAP = {
    0: np.array([1.0, 1.0, 0.0]),
    1: np.array([0.0, 0.5, 0.0]),
    2: np.array([0.0, 1.5, 0.0]),
    3: np.array([2.0, 0.5, 0.0]),
    4: np.array([2.0, 1.5, 0.0])
}
DRONE_ID = 10
# ==========================================

def main(input_file=None):
    # --- CSV保存の準備 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # logsフォルダに保存するように修正
    save_dir = os.path.join(current_dir, "..", "logs")
    os.makedirs(save_dir, exist_ok=True)
    
    # ファイル名に日付を入れる
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"result_{now_str}.csv"
    csv_path = os.path.join(save_dir, csv_filename)
    
    # CSVファイルを開く
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp(ms)", "X(m)", "Y(m)", "Z(m)"])
    print(f"CSV保存先: {csv_path}")

    # --- RealSense設定 ---
    pipeline = rs.pipeline()
    config = rs.config()

    if input_file:
        rs.config.enable_device_from_file(config, input_file, repeat_playback=False)
        print(f"File Playback Mode: {input_file}")
    else:
        # ライブ設定 (赤外線モード)
        config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        print("Live Camera Mode (IR)")

    profile = pipeline.start(config)

    # ライブ時のみEmitterをOFF
    if not input_file:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0.0)

    # カメラパラメータ取得
    stream_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    intrinsics = stream_profile.get_intrinsics()
    
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intrinsics.coeffs)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    try:
        while True:
            # 【修正点】再生終了時(エラー発生時)にループを抜ける処理を追加
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("再生が終了しました。")
                break
            
            ir_frame = frames.get_infrared_frame(1)
            depth_frame = frames.get_depth_frame()

            if not ir_frame or not depth_frame:
                continue
            
            timestamp = frames.get_timestamp()
            ir_image = np.asanyarray(ir_frame.get_data())
            display_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)

            corners, ids, rejected = cv2.aruco.detectMarkers(display_image, aruco_dict, parameters=parameters)

            if ids is not None:
                ids = ids.flatten()
                
                # --- カメラ位置推定 ---
                obj_points = []
                img_points = []

                for i, marker_id in enumerate(ids):
                    if marker_id in REF_MARKERS_MAP:
                        c = corners[i][0]
                        img_points.append([(c[0][0] + c[2][0]) / 2.0, (c[0][1] + c[2][1]) / 2.0])
                        obj_points.append(REF_MARKERS_MAP[marker_id])

                T_world_cam = None

                if len(obj_points) > 0:
                    ref_id = 0 if 0 in ids else ids[list(ids).index(next(id for id in ids if id in REF_MARKERS_MAP))]
                    ref_idx = list(ids).index(ref_id)
                    
                    rvec_m, tvec_m, _ = cv2.aruco.estimatePoseSingleMarkers(corners[ref_idx], MARKER_SIZE_REF, camera_matrix, dist_coeffs)
                    rvec_m = rvec_m[0]
                    tvec_m = tvec_m[0]
                    
                    R_m, _ = cv2.Rodrigues(rvec_m)
                    T_cam_marker = np.eye(4)
                    T_cam_marker[:3, :3] = R_m
                    T_cam_marker[:3, 3] = tvec_m.flatten()
                    
                    T_marker_world = np.eye(4)
                    T_marker_world[:3, 3] = REF_MARKERS_MAP[ref_id]
                    
                    T_world_cam = np.dot(T_marker_world, np.linalg.inv(T_cam_marker))

                # --- ドローン位置推定 & CSV保存 ---
                if DRONE_ID in ids and T_world_cam is not None:
                    drone_idx = list(ids).index(DRONE_ID)
                    c_d = corners[drone_idx][0]
                    u = int((c_d[0][0] + c_d[2][0]) / 2)
                    v = int((c_d[0][1] + c_d[2][1]) / 2)
                    
                    dist_drone = depth_frame.get_distance(u, v)
                    
                    if dist_drone > 0:
                        p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], dist_drone)
                        p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
                        p_world = np.dot(T_world_cam, p_cam_h)
                        
                        x_w, y_w, z_w = p_world[0], p_world[1], p_world[2]
                        
                        csv_writer.writerow([timestamp, x_w, y_w, z_w])
                        
                        text = f"Drone: X={x_w:.2f}m, Y={y_w:.2f}m, H={z_w:.2f}m"
                        cv2.putText(display_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        print(text)
                        
                        cv2.drawMarker(display_image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)

            cv2.imshow('Drone Tracker (IR)', display_image)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if input_file and key == 32: # Space key
                cv2.waitKey()

    finally:
        csv_file.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("終了しました。CSV保存完了。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to .bag file")
    args = parser.parse_args()
    main(args.file)