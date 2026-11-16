import cv2           # opencvを読み込み
import numpy         # たくさんの数値を配列にしてまとめて計算できるPythonライブラリ
import time          # time.sleep関数を使用するため
import pyrealsense2  # PythonからRealSenseを操作するためのライブラリを読み込み

clicked_point = None
def on_mouse(event, x, y, flags, param): 
    global clicked_point              # global は、関数の中から、そのファイルの一番外側で定義した変数に代入するための宣言
    if event == cv2.EVENT_LBUTTONDOWN:# cv2.EVENT_LBUTTONDOWN は、マウスの左ボタンを押した瞬間を表すOpenCVの定数
        clicked_point = (x, y)

pipeline = pyrealsense2.pipeline() # pipeline はカメラからデータを受け取るためのオブジェクトを作成
config = pyrealsense2.config()     # どの種類のデータを、どんな設定で受け取るかを指定するオブジェクトを作成
config.enable_stream(pyrealsense2.stream.depth, 640, 480, pyrealsense2.format.z16, 30) # 深度画像・幅640・高さ480・16ビット形式・毎秒30フレームの順で指定
config.enable_stream(pyrealsense2.stream.color, 640, 480, pyrealsense2.format.bgr8, 30)# stream.color はカラー映像、`bgr8` は青・緑・赤をそれぞれ8ビットで持つ形式
pipeline.start(config)             # config の設定を使って、カメラからのデータ受信を開始
align_to_color = pyrealsense2.align(pyrealsense2.stream.color) # stream.color はカラー画像の座標を基準にするという指定

try:
    cv2.namedWindow("Color")                # ウィンドウの作成
    cv2.setMouseCallback("Color", on_mouse) # マウス操作があったらon_mouse関数を呼び出す

    while True:
        frames = pipeline.wait_for_frames() # 新しいフレームのまとまりが届くまで待つ、5秒でタイムアウト

        aligned_frames = align_to_color.process(frames) # 深度データをカラー画像の座標に対応づけ

        #depth = frames.get_depth_frame()    # 届いたフレームのまとまりから、深度画像を取り出し
        #color = frames.get_color_frame()    # 届いたフレームのまとまりから、カラー画像を取り出し


        # if color:
        #     color_image = numpy.asanyarray(color.get_data())
        #     cv2.imshow("Color", color_image)# stream.color はカラー映像、`bgr8` は青・緑・赤をそれぞれ8ビットで持つ形式

        # if not depth:
        #     continue
        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()

        if not depth or not color:
            continue

        color_image = numpy.asanyarray(color.get_data()).copy()   # .copy()で独立したコピーを作り、そのコピーに丸や文字を書き込み
        
        if clicked_point is None:
            x = color.get_width() // 2 
            y = color.get_height() // 2
        else:
            x, y = clicked_point

        distance = depth.get_distance(x, y)                       # その座標の1ピクセルの深度をメートルで返す関数

        text = f"{distance:.2f} m" if distance > 0 else "No depth"# JavaScriptの 条件 ? 値A : 値B に近く、Pythonでは 値A if 条件 else 値B と書く
        cv2.circle(color_image, (x, y), 5, (0, 255, 0), 2)        # color_image の (x, y) に、半径5ピクセル、線の太さ2ピクセルの丸を描く
        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)# 同じ画像に text を書き込み、(10, 30)`は文字の描画位置。後ろの引数は順に フォント、文字の大きさ、色、線の太さ
        cv2.imshow("Color", color_image)                          # 丸と文字を書き込んだ画像を Color ウィンドウに表示

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