import cv2
import shutil
import os
import datetime

def get_user_setting ():
  f = open('./setting.txt', 'r') # 設定ファイルの読み込み
  setting_txt_list = f.readlines()
  result_list = {
    "img_path": setting_txt_list[16][:len(setting_txt_list[16])-1],
    "interval_update": float(setting_txt_list[19][:len(setting_txt_list[19])-1]),
    "interval_make_img": float(setting_txt_list[22][:len(setting_txt_list[22])-1]),
    "is_show_movie": bool(setting_txt_list[25][:len(setting_txt_list[25])-1] == "y")
  }
  return result_list

if __name__ == "__main__":
  user_setting_list = get_user_setting()
  dt1 = datetime.datetime.now()
  update_time = datetime.datetime.now()
  photo_id = 0
  is_face_flag = False
  judge_time = datetime.datetime.now()
  if len(user_setting_list["img_path"]) != 0 and user_setting_list["img_path"] != "." and user_setting_list["img_path"][len(user_setting_list["img_path"])-1:] != "/":
    try:
      shutil.rmtree(user_setting_list["img_path"])
      os.mkdir(user_setting_list["img_path"])
    except:
      os.mkdir(user_setting_list["img_path"])
  else:
    raise ValueError('Do not specify the path each "{}"!!'.format(user_setting_list["img_path"]))
  # 内蔵カメラを起動
  cap = cv2.VideoCapture(0)

  # OpenCVに用意されている顔認識するためのxmlファイルのパス
  cascade_path = "./cascadefiles/haarcascade_frontalface_default.xml"
  # カスケード分類器の特徴量を取得する
  cascade = cv2.CascadeClassifier(cascade_path)
  # 顔に表示される枠の色を指定（青色）
  color = (255,0,0)

  while True:
    # qキーを押すとループ終了
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
      break

    if (datetime.datetime.now() - update_time).seconds < user_setting_list["interval_update"]:
      continue
    # 内蔵カメラから読み込んだキャプチャデータを取得
    ret, frame = cap.read()

    # 顔認識の実行
    facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10,10))

    # 顔が見つかったらcv2.rectangleで顔に白枠を表示する
    if len(facerect) > 0:
      if not is_face_flag:
        judge_time = datetime.datetime.now() # ジャッジする時間を計測
        is_face_flag = True # 顔認識のジャッジを開始
      for rect in facerect:
        img = cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)
      if (datetime.datetime.now() - dt1).seconds >= user_setting_list["interval_make_img"]: # もしも切り取り間隔なら
        if (update_time - judge_time).seconds >= 5 and is_face_flag: # 顔認識のジャッジが成功すれば
          cv2.imwrite('{}/result_{}.png'.format(user_setting_list["img_path"],photo_id), img)
          photo_id += 1
          dt1 = datetime.datetime.now()
    else:
      is_face_flag = False # 顔認識のジャッジの初期化
    # 表示
    if user_setting_list["is_show_movie"]:
      cv2.imshow("frame test", frame)
    update_time = datetime.datetime.now()


  # 内蔵カメラを終了
  cap.release()
  cv2.destroyAllWindows()