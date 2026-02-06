import pyrealsense2 as rs
import numpy as np
import cv2
import argparse

# ==========================================
# ユーザー設定エリア
# ==========================================
# マーカーの物理サイズ [メートル]
MARKER_SIZE_REF = 0.15   # 床のマーカーのサイズ
MARKER_SIZE_DRONE = 0.10 # ドローンのマーカーのサイズ

# 基準マーカーの世界座標定義 {ID: (X, Y, Z)}
# Zは床面なので常に0.0
REF_MARKERS_MAP = {
    0: np.array([1.0, 1.0, 0.0]), # 原点基準
    1: np.array([0.0, 0.5, 0.0]),
    2: np.array([0.0, 1.5, 0.0]),
    3: np.array([2.0, 0.5, 0.0]),
    4: np.array([2.0, 1.5, 0.0])
}
DRONE_ID = 10
# ==========================================

def main(input_file=None):
    # パイプライン設定
    pipeline = rs.pipeline()
    config = rs.config()

    if input_file:
        # ファイルから読み込む場合
        rs.config.enable_device_from_file(config, input_file, repeat_playback=False)
        print(f"File Playback Mode: {input_file}")
    else:
        # ライブカメラの場合
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        print("Live Camera Mode")

    # ストリーミング開始
    profile = pipeline.start(config)
    
    # 深度とカラーの位置合わせ用
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 内部パラメータ取得
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # OpenCV用のカメラ行列(K)と歪み係数(dist)を作成
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array(intrinsics.coeffs)

    # ArUco設定
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # マーカー検出
            corners, ids, rejected = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)

            if ids is not None:
                ids = ids.flatten()
                
                # --- 1. 基準マーカーを使ってカメラの位置(Pose)を特定する ---
                obj_points = [] # 世界座標 (3D)
                img_points = [] # 画像座標 (2D)

                # 見えているマーカーのうち、基準マーカーマップにあるものを集める
                for i, marker_id in enumerate(ids):
                    if marker_id in REF_MARKERS_MAP:
                        # 画像上のコーナー座標（中心を使用）
                        c = corners[i][0]
                        center_x = (c[0][0] + c[2][0]) / 2.0
                        center_y = (c[0][1] + c[2][1]) / 2.0
                        img_points.append([center_x, center_y])
                        
                        # 対応する世界座標
                        obj_points.append(REF_MARKERS_MAP[marker_id])

                # 少なくとも1つ以上の基準マーカーが見えている場合
                if len(obj_points) > 0:
                    obj_points = np.array(obj_points, dtype=np.float32)
                    img_points = np.array(img_points, dtype=np.float32)

                    # SolvePnPで「世界」に対する「カメラ」の位置姿勢を計算
                    # ※マーカーが平面配置なのでSOLVEPNP_IPPEなどが精度良いが、汎用的にITERATIVEを使用
                    # ※ポイントが少ない(1-3個)場合でも動くように工夫
                    if len(obj_points) >= 4:
                        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                    else:
                        # 点が少ない場合は推測精度が落ちるが計算は可能 (1点でもArUcoのサイズ情報があればPose推定可能だが、
                        # ここでは簡易的に「複数点PnP」ロジックを通すため、estimatePoseSingleMarkersの結果を使っても良い。
                        # 今回はシンプルにするため、最も信頼できるマーカー1つで代表させるか、
                        # リスト化してsolvePnPに投げる（OpenCVのsolvePnPは4点未満だとエラーになる場合があるため注意）
                        
                        # ★ロバスト化: 点が少ない場合は、最初に見つかった基準マーカー単体からPoseを出す
                        target_idx = list(ids).index(next(ID for ID in ids if ID in REF_MARKERS_MAP))
                        rvec_s, tvec_s, _ = cv2.aruco.estimatePoseSingleMarkers(corners[target_idx], MARKER_SIZE_REF, camera_matrix, dist_coeffs)
                        rvec = rvec_s[0]
                        tvec = tvec_s[0]
                        
                        # そのマーカーの世界座標分だけオフセットを考慮する必要がある
                        # (単体Pose推定は「そのマーカー原点」基準になるため、世界座標への変換行列補正が必要)
                        # ここは少し複雑になるため、今回は「4点以上設置されている前提」で
                        # 万が一隠れてもsolvePnPが使えるよう、2点以上見えている状態推奨のロジックにします。
                        # ※厳密には： T_cam_world = T_cam_marker * T_marker_world
                        
                        # 簡易実装: PnPが使えないときはスキップ（または前回の値を保持）
                        success = False
                        if len(obj_points) >= 4: # 本来は4点以上推奨
                             pass 
                    
                    # 4点未満でもsolvePnPを通すためのフラグ (SOLVEPNP_ITERATIVEは4点以上必要。SQPNPなどはより少ない点で可)
                    # 今回はユーザー構成が5点あるので、見えている点を全て使います。
                    if len(obj_points) >= 1:
                        # 複数のマーカー中心点だけだと回転が決まりにくいので、コーナー点全てを使うのがベストですが、
                        # 実装が複雑になるため、今回は「estimatePoseSingleMarkers」で見えたマーカー個別に
                        # 座標変換行列を作り、平均を取るアプローチの方が実装ミスが少ないです。
                        
                        # --- 簡易・堅牢版ロジック ---
                        # 1. 見えている基準マーカー1つ選ぶ（ID0を優先、なければ最初に見えたもの）
                        ref_id = 0 if 0 in ids else ids[list(ids).index(next(id for id in ids if id in REF_MARKERS_MAP))]
                        ref_idx = list(ids).index(ref_id)
                        
                        # 2. そのマーカーに対するカメラの相対位置を出す
                        rvec_m, tvec_m, _ = cv2.aruco.estimatePoseSingleMarkers(corners[ref_idx], MARKER_SIZE_REF, camera_matrix, dist_coeffs)
                        rvec_m, tvec_m = rvec_m[0], tvec_m[0] # 次元削減
                        
                        # 3. 座標変換行列を作成: T_cam_marker
                        R_m, _ = cv2.Rodrigues(rvec_m)
                        T_cam_marker = np.eye(4)
                        T_cam_marker[:3, :3] = R_m
                        T_cam_marker[:3, 3] = tvec_m.T
                        
                        # 4. マーカーから世界への変換: T_marker_world (平行移動のみと仮定 ※回転していなければ)
                        # もし床のマーカーが回転して貼られている場合はここも回転が必要
                        world_pos = REF_MARKERS_MAP[ref_id]
                        T_marker_world = np.eye(4)
                        T_marker_world[:3, 3] = world_pos # 平行移動
                        
                        # ※注意: estimatePoseSingleMarkersは「マーカー中心が原点」。
                        # カメラ位置を知りたいわけではなく、「ドローンのカメラ座標」を「世界座標」にしたい。
                        # 変換式: P_world = T_marker_world * (T_cam_marker)^-1 * P_cam
                        
                        T_world_cam = np.dot(T_marker_world, np.linalg.inv(T_cam_marker))

                        # --- 2. ドローンの座標を計算 ---
                        if DRONE_ID in ids:
                            drone_idx = list(ids).index(DRONE_ID)
                            c_d = corners[drone_idx][0]
                            u = int((c_d[0][0] + c_d[2][0]) / 2)
                            v = int((c_d[0][1] + c_d[2][1]) / 2)
                            
                            # 深度取得
                            dist_drone = depth_frame.get_distance(u, v)
                            
                            if dist_drone > 0:
                                # カメラ座標系でのドローン位置 (X, Y, Z)
                                p_cam = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], dist_drone)
                                p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]) # 同次座標
                                
                                # 世界座標へ変換
                                p_world = np.dot(T_world_cam, p_cam_h)
                                
                                x_w, y_w, z_w = p_world[0], p_world[1], p_world[2]
                                
                                # 結果表示
                                text = f"Drone: X={x_w:.2f}m, Y={y_w:.2f}m, H={z_w:.2f}m"
                                cv2.putText(color_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                print(text)
                                
                                # 座標軸の描画 (ドローンの位置に)
                                cv2.drawMarker(color_image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # 画面表示
            cv2.imshow('Drone Tracker', color_image)
            
            # 再生モードでのキー操作 (スペースで一時停止、qで終了)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            if input_file and key == 32: # Space key
                cv2.waitKey() # Resume waiting for key

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 引数があれば録画ファイルを再生、なければライブカメラ
    parser.add_argument("--file", type=str, help="Path to .bag file for playback")
    args = parser.parse_args()
    
    main(args.file)