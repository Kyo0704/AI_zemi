import cv2
import time
import mediapipe as mp
import shutil
import os
import augmentation_save as aus
import glob
 
# カメラ解像度の設定
wCam, hCam = 224, 224
 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
 
# カメラが1台のみ接続されている場合は0を指定。
# 2台以上接続されている場合は、カメラIDを指定。
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)



count = 0
print("名前を入力してください")
name = str(input())
mpath = '{}'.format(name)

if os.path.isdir(mpath) :
    shutil.rmtree(mpath)
    
os.mkdir(mpath)
#時間測定開始
t = time.time()

def read_dir(target_path):
    # ディレクトリ内の全ての.pngファイルを取得
    files = glob.glob(f"{target_path}/*.png")
    # 各ファイルを読み込み、水増しをする
    for f in files:
        aus.save_dg(f)
 
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
 
        if success==False:
            continue
        
        # パオ―マンス向上のため、オプションで参照渡しの画像を書き込み不可にする。
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
        # 画像を保存する
        imagefile = "image{}.png".format(count)
        cv2.imwrite(imagefile, image)
        count += 1
        
        if results.detections:
            new_path = shutil.move(imagefile, mpath)
        #20秒たったら終了
        c = time.time()
        if c - t >= 20:
            break
# 後始末。しなくても終わる。
cap.release()
cv2.destroyAllWindows()
read_dir(mpath)