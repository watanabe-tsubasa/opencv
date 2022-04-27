import cv2 as cv

# VideoCapture オブジェクトを取得します
capture0 = cv.VideoCapture(0)
capture1 = cv.VideoCapture(1)

while(True):
    ret0, frame0 = capture0.read()
    ret1, frame1 = capture1.read()

    cv.imshow('frame0',frame0)
    cv.imshow('frame1',frame1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture0.release()
capture1.release()
cv.destroyAllWindows()