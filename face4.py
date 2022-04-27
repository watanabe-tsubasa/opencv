import sys
import cv2 
import time
import csv

f = open("cl_save.csv","a")
csvWriter = csv.writer(f)
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
start0 = None
start1 = None
end0 = 0
end1 = 1
elapsed_time0 = 0
elapsed_time1 = 1

if cap0.isOpened() is False:
    print("can not open camera0")
    sys.exit()
if cap1.isOpened() is False:
    print("can not open camera1")
    sys.exit()

# 評価器を読み込み
cascade = cv2.CascadeClassifier('opencv-4/data/haarcascades/haarcascade_frontalface_alt2.xml')

# Webカメラの映像に対して延々処理を繰り返すためwhile Trueで繰り返す。
while True:
    # VideoCaptureから1フレーム読み込む
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    # そのままの大きさだと処理速度がきついのでリサイズ
    # frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))

    # 処理速度を高めるために画像をグレースケールに変換したものを用意
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # 顔検出 detecctMultiScale()は検出器のルール(cascade)に従って検出した結果をfacerectに返す関数
    facerect0 = cascade.detectMultiScale(
        gray0,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )
    
    facerect1 = cascade.detectMultiScale(
        gray1,
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )

    if len(facerect0) != 0:
        if start0 is None:
            start0 = time.time()
            print(f'dif:{start0-end0}')
            for x, y, w, h in facerect0:
                # 顔の部分
                face_gray0 = gray0[y: y + h, x: x + w]
                # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
                show_face_gray0 = cv2.resize(face_gray0, (int(gray0.shape[1]), int(gray0.shape[0])))
                cv2.imshow('face0', show_face_gray0)
                # imshow()で見たい画像を表示する
                # 顔検出した部分に枠を描画
                cv2.rectangle(
                    frame0,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness=2
                )
        #検出に間が空いた場合、別の顧客とみなして結果を出力する
            if start0 - end0 > 1 and elapsed_time0 != 0 :
                print(elapsed_time0)
                listData0 = []
                listData0.append(str(elapsed_time0))
                csvWriter.writerow(listData0)
                elapsed_time0 = 0
        else :
            for x, y, w, h in facerect0:
                # 顔の部分
                face_gray0 = gray0[y: y + h, x: x + w]
                # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
                show_face_gray0 = cv2.resize(face_gray0, (int(gray0.shape[1]), int(gray0.shape[0])))
                cv2.imshow('face0', show_face_gray0)
                # 顔検出した部分に枠を描画
                cv2.rectangle(
                    frame0,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness=2
                )
        
    elif len(facerect0) == 0 and start0 is not None :
        end0 = time.time()
        elapsed_time0 = elapsed_time0 + end0 - start0
        #測定された時間を暫定的にprint
        print(f'add:{end0-start0},ela:{elapsed_time0}')
        start0 = None

    cv2.imshow('frame0', frame0)

    if len(facerect1) != 0:
        if start1 is None:
            start1 = time.time()
            print(f'dif:{start1-end1}')
            for x, y, w, h in facerect1:
                # 顔の部分
                face_gray1 = gray1[y: y + h, x: x + w]
                # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
                show_face_gray1 = cv2.resize(face_gray1, (int(gray1.shape[1]), int(gray1.shape[0])))
                cv2.imshow('face1', show_face_gray1)
                # imshow()で見たい画像を表示する
                # 顔検出した部分に枠を描画
                cv2.rectangle(
                    frame1,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness=2
                )
        #検出に間が空いた場合、別の顧客とみなして結果を出力する
            if start1 - end1 > 1 and elapsed_time1 != 0 :
                print(elapsed_time1)
                listData1 = []
                listData1.append(str(elapsed_time1))
                csvWriter.writerow(listData1)
                elapsed_time1 = 0
        else :
            for x, y, w, h in facerect1:
                # 顔の部分
                face_gray1 = gray1[y: y + h, x: x + w]
                # くり抜いた顔の部分を表示(処理には必要ない。ただ見たいだけ。)
                show_face_gray1 = cv2.resize(face_gray1, (int(gray1.shape[1]), int(gray1.shape[0])))
                cv2.imshow('face1', show_face_gray1)
                # 顔検出した部分に枠を描画
                cv2.rectangle(
                    frame1,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 255),
                    thickness=2
                )
        
    elif len(facerect1) == 0 and start1 is not None :
        end1 = time.time()
        elapsed_time1 = elapsed_time1 + end1 - start1
        #測定された時間を暫定的にprint
        print(f'add:{end1-start1},ela:{elapsed_time1}')
        start1 = None

    cv2.imshow('frame1', frame1)

    # キー入力を1ms待って、k が27（ESC）だったら結果を出力してBreakする
    k = cv2.waitKey(1)
    if k == 27:
        print(elapsed_time0)
        print(elapsed_time1)
        listData0 = []
        listData1 = []
        listData0.append(str(elapsed_time0))
        listData1.append(str(elapsed_time1))
        csvWriter.writerow(listData0)
        csvWriter.writerow(listData1)
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap0.release()
cap1.release()
cv2.destroyAllWindows()