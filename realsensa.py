import cv2           # opencvを読み込み
import numpy         # たくさんの数値を配列にしてまとめて計算できるPythonライブラリ
import time          # time.sleep関数を使用するため
import pyrealsense2  # PythonからRealSenseを操作するためのライブラリを読み込み

pipeline = pyrealsense2.pipeline() # pipeline はカメラからデータを受け取るためのオブジェクトを作成
config = pyrealsense2.config()     # どの種類のデータを、どんな設定で受け取るかを指定するオブジェクトを作成
config.enable_stream(pyrealsense2.stream.depth, 640, 480, pyrealsense2.format.z16, 30) # 深度画像・幅640・高さ480・16ビット形式・毎秒30フレームの順で指定
config.enable_stream(pyrealsense2.stream.color, 640, 480, pyrealsense2.format.bgr8, 30)# stream.color はカラー映像、`bgr8` は青・緑・赤をそれぞれ8ビットで持つ形式
pipeline.start(config)             # config の設定を使って、カメラからのデータ受信を開始

try:
    while True:
        frames = pipeline.wait_for_frames() # 新しいフレームのまとまりが届くまで待つ、5秒でタイムアウト
        depth = frames.get_depth_frame()    # 届いたフレームのまとまりから、深度画像を取り出し

        color = frames.get_color_frame()
        if color:
            color_image = numpy.asanyarray(color.get_data())
            cv2.imshow("Color", color_image)# stream.color はカラー映像、`bgr8` は青・緑・赤をそれぞれ8ビットで持つ形式

        if not depth:
            continue

        # x = depth.get_width()  // 2         # 画像の幅を半分にして、中央の横座標を求める
        # y = depth.get_height() // 2         # 画像の高さを半分にして、中央の縦座標を求める
        # distance = depth.get_distance(x, y) # (x, y) にある1ピクセルの深度をメートルで取得

        # if distance > 0:
        #     print(f"中央の深度: {distance:.2f} m")
        # else:
        #     print("中央の深度: 測定できません")

        # time.sleep(0.2)                     # 0.2秒待ち

        raw_depth = numpy.asanyarray(depth.get_data()) # RealSenseから全画素の深度データを取り出し、NumPy配列として扱う
                                                       # 幅640、高さ480なら、画面の各位置に対応する 320,240個の数値があり、配列に 
        depth_m = raw_depth * depth.get_units()        # カメラの数値が1増えると、実際の奥行きが何メートル増えるかを返すので生の値をメートルに変換

        scaled = numpy.clip(depth_m / 4.0, 0, 1)       # 色分けする範囲を0〜4 mに設定
        depth_8bit = (scaled * 255).astype(numpy.uint8)# 0〜1 の値に255を掛けて、OpenCVの色付けに使う 0〜255 の整数に変換

        depth_map = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO) # 配色設定
        mask = raw_depth == 0
        depth_map[mask] = (0, 0, 0)                       # raw_depth == 0 で深度を測れなかった画素を選び、表示色を黒に

        cv2.imshow("Depth map", depth_map)                # Depth map という名前のウィンドウに、今の色付き画像を表示
        if cv2.waitKey(1) == ord("q"):
            break

except KeyboardInterrupt: # Ctrl+C による中断を受け取る
    pass
finally:
    pipeline.stop()       #ストリームを停止し、カメラを解放