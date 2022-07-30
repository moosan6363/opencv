import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像ファイルを読み込む
img = cv2.imread("train.png")

# 画像を正規化
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# 1. ヒストグラム平坦化
# result = cv2.equalizeHist(v)

# 2. 適応的ヒストグラム平坦化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
result = clahe.apply(v)

# 明度のヒストグラム
plt.hist([v.ravel(), result.ravel()],256,[0,256]);plt.show()

# カラー画像に戻す
hsv = cv2.merge((h,s,result))
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# 画像をグレースケールへ変換
img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

# カスケードファイルのパス
cascade_path = "cascade/cascade.xml"
# cascade_path = "haarcascades/haarcascade_frontalface_alt.xml"
# cascade_path = "haarcascades/haarcascade_eye.xml"
# カスケード分類器の特徴量取得
cascade = cv2.CascadeClassifier(cascade_path)
# 顔認識
faces=cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(100,100))

# 検出位置描画
for x,y,w,h in faces:
    cv2.rectangle(rgb, (x,y), (x+w, y+h), (0, 0, 255), thickness=30)

# 顔検出画像表示
plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
plt.show()
# 顔検出画像出力
# cv2.imwrite("out.jpg", img)