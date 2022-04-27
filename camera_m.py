import cv2

i = 0
flag = True
captures = []

# VideoCapture オブジェクトを取得します
while( flag ):
    capture = cv2.VideoCapture(i)
    ret, frame = capture.read()
    flag = ret
    if flag:
        i += 1
        captures.append( capture )

while(True):
    for i, capture in enumerate( captures ):
        ret, frame = capture.read()
        cv2.imshow( f'frame{i}', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()