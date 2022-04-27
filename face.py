import sys
import cv2 # OpenCV のインポート

'''
参考
@link http://ensekitt.hatenablog.com/entry/2017/12/19/200000
@link https://note.nkmk.me/python-opencv-face-detection-haar-cascade/
@link https://note.nkmk.me/python-opencv-mosaic/
@link http://workpiles.com/2015/04/opencv-detectmultiscale-scalefactor/
'''

# cv2.VideoCapture() で VideoCaptureのインスタンスを作成する。
# 引数にはコンピュータに接続されているカメラの番号を指定。
# コンピュータにカメラが1台だけしか接続されていない場合には「0」を指定すればよい。
# もし複数のカメラが接続されている場合は「1」などの番号を指定します。
# 参考：https://weblabo.oscasierra.net/python/opencv-videocapture-camera.html
cap = cv2.VideoCapture(0)

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# 評価器を読み込み
# https://github.com/opencv/opencv/tree/master/data/haarcascades
cascade = cv2.CascadeClassifier('opencv-4/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('opencv-4/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

# モザイク処理の関数。やってることは、取得した画像を拡大した後、もとのサイズに縮小する
# 参考：https://note.nkmk.me/python-opencv-mosaic/
def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST) # おっきくしてから…
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST) # 元のサイズに戻す。

# モザイクの範囲を指定する関数
def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

# Webカメラの映像に対して延々処理を繰り返すためwhile Trueで繰り返す。
while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # そのままの大きさだと処理速度がきついのでリサイズ
    frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))

    # 処理速度を高めるために画像をグレースケールに変換したものを用意
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出 detecctMultiScale()は検出器のルール(cascade)に従って検出した結果をfacerectに返す関数
    facerect = cascade.detectMultiScale(
        gray,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )

    if len(facerect) != 0:
        for x, y, w, h in facerect:
            # 顔の部分(この顔の部分に対して目の検出をかける)
            face_gray = gray[y: y + h, x: x + w]

            # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
            show_face_gray = cv2.resize(face_gray, (int(gray.shape[1]), int(gray.shape[0])))
            cv2.imshow('face', show_face_gray)
            # imshow()で見たい画像を表示する

            # 顔の部分から目の検出
            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.11, # ここの値はPCのスペックに依存するので適宜修正してください
                minNeighbors=3,
                minSize=(15, 15)
            )

            if len(eyes) == 0:
                # 目が閉じられたとみなす
                cv2.putText(
                    frame,
                    'close your eyes',
                    (x, y - 10), # 位置を少し調整
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255,0),
                    2,
                    cv2.LINE_AA
                )
            else:
                for (ex, ey, ew, eh) in eyes:
                    # 目の部分にモザイク処理
                    frame = mosaic_area(
                        frame,
                        int((x + ex) - ew / 2),
                        int(y + ey),
                        int(ew * 2.5),
                        eh
                    )

            # 顔検出した部分に枠を描画
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                thickness=2
            )

    cv2.imshow('frame', frame)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()