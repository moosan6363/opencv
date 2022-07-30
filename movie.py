import cv2
import os
import sys
import time

if __name__ == "__main__" :

    cap = cv2.VideoCapture("introduce.mp4")
    if not cap.isOpened() :
        sys.exit()

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter('detect_face.mp4',fourcc, fps, (cap_width, cap_height))

    # cascade_base_path = "haarcascades/"
    # #準備　カスケードを取得
    # face_cascade = cv2.CascadeClassifier(os.path.join(cascade_base_path, 'haarcascade_frontalface_alt_tree.xml'))
    # right_eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_base_path, 'haarcascade_righteye_2splits.xml'))
    # left_eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_base_path, 'haarcascade_lefteye_2splits.xml'))

    face_cascade = cv2.CascadeClassifier("cascade/cascade.xml")
    start = time.time()

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

            # 1. フレームの中に顔が写っている　
            face_points = face_cascade.detectMultiScale(img_gray)

            for (fx,fy,fw,fh) in face_points:

            #     #2. ROI(Region of Interest:対象領域)となる画像を切り出す
            #     #右領域と左領域でそれぞれ分割(アシュラ男爵方式)
            #     width_center = fx + int(fw * 0.5)
            #     face_right_gray = img_gray[fy:fy+fh, fx:width_center]
            #     face_left_gray = img_gray[fy:fy+fh, width_center:fx+fw]

            #     #3. 右目と左目の両方が写っているか判定し出力
            #     right_eye_points = right_eye_cascade.detectMultiScale(face_right_gray)
            #     left_eye_points = left_eye_cascade.detectMultiScale(face_left_gray)

            #     if 0 < len(right_eye_points) :
            #         #右目はオレンジ
            #         (rx,ry,rw,rh) = right_eye_points[0]
            #         cv2.rectangle(frame,(fx+rx,fy+ry),(fx+rx+rw,fy+ry+rh),(0,255,255),2)

            #     if 0 < len(left_eye_points) :
            #         #左目は赤
            #         (lx,ly,lw,lh) = left_eye_points[0]
            #         cv2.rectangle(frame,(width_center+lx,fy+ly),(width_center+lx+lw,fy+ly+lh),(0,0,255),2)

                #顔全体は緑
                cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)

            writer.write(frame) 
        else : break

    print("処理時間 {} 秒".format(time.time() - start))
    writer.release()
    cap.release()