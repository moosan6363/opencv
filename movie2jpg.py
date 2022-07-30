import cv2
import os
import sys
import time

if __name__ == "__main__" :

    cap = cv2.VideoCapture("train.mp4")
    if not cap.isOpened() :
        sys.exit()

    start = time.time()

    cnt = 0

    while True :
        # 1フレーム読み込み
        ret, frame = cap.read()

        if ret :
            # 現在のフレームを出力
            print("Processing frame: {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            # 画像を適応的ヒストグラム平坦化
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
            result = clahe.apply(v)
            hsv = cv2.merge((h,s,result))
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # グレースケールに変換
            img_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            
            if cnt%2 == 0 :
                filePath = "ok/{}.jpg".format(str(int(cnt/2)).zfill(4))
                cv2.imwrite(filePath, rgb)
            cnt += 1
       
        else : break

    print("処理時間 {} 秒".format(time.time() - start))
    cap.release()