import sys
import cv2 
import time
import csv

f = open("cl_save.csv","a")
csvWriter = csv.writer(f)
cap = cv2.VideoCapture(0)
start = None
end = 0
elapsed_time = 0

if cap.isOpened() is False:
    print("can not open camera")
    sys.exit()

# 評価器を読み込み
cascade = cv2.CascadeClassifier('opencv-4/data/haarcascades/haarcascade_frontalface_alt2.xml')

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
        if start is None:
            start = time.time()
            print(f'dif:{start-end}')
            for x, y, w, h in facerect:
                # 顔の部分
                face_gray = gray[y: y + h, x: x + w]
                # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
                show_face_gray = cv2.resize(face_gray, (int(gray.shape[1]), int(gray.shape[0])))
                cv2.imshow('face', show_face_gray)
                # imshow()で見たい画像を表示する
                # 顔検出した部分に枠を描画
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness=2
                )
        #検出に間が空いた場合、別の顧客とみなして結果を出力する
            if start - end > 1 and elapsed_time != 0 :
                print(elapsed_time)
                listData = []
                listData.append(str(elapsed_time))
                csvWriter.writerow(listData)
                elapsed_time = 0
        else :
            for x, y, w, h in facerect:
                # 顔の部分
                face_gray = gray[y: y + h, x: x + w]
                # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
                show_face_gray = cv2.resize(face_gray, (int(gray.shape[1]), int(gray.shape[0])))
                cv2.imshow('face', show_face_gray)
                # 顔検出した部分に枠を描画
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness=2
                )
        
    elif len(facerect) == 0 and start is not None :
        end = time.time()
        elapsed_time = elapsed_time + end - start
        #測定された時間を暫定的にprint
        print(f'add:{end-start},ela:{elapsed_time}')
        start = None

    cv2.imshow('frame', frame)

    # キー入力を1ms待って、k が27（ESC）だったら結果を出力してBreakする
    k = cv2.waitKey(1)
    if k == 27:
        print(elapsed_time)
        listData = []
        listData.append(str(elapsed_time))
        csvWriter.writerow(listData)
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()