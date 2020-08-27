import matplotlib
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import re
import os
import time
import datetime
import requests
import json
import numpy as np


apiURL = 'http://iot.cht.com.tw/iot/v1/device/23632626264/rawdata'  #17750119494
headers = { "CK":"PK4ACRESYYKAKUUTPE","Content-Type": "application/json",}  #PK0B9PPBEHGEA4ASC7




#載入模型h5檔案
model = load_model("C:\\Last_test\\DL\\sick2.h5")
model.summary()
#規範化圖片大小和畫素值
def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (150, 150))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input一張圖片
    pre_x = np.array(pre_x) / 255.0
    return pre_x

#要預測的圖片儲存在這裡
predict_dir = 'C:\\Last_test\\DL\\MAIN\\AI_TEST'
#這個路徑下有兩個檔案，分別是cat和dog
test = os.listdir(predict_dir)
#列印後：['cat', 'dog']
print(test)
#新建一個列表儲存預測圖片的地址
images = []
fdd = []
#獲取每張圖片的地址，並儲存在列表images中
 
for testpath in test:
    for fn in os.listdir(os.path.join(predict_dir, testpath)):
        if fn.endswith('jpg'):
            fd = os.path.join(predict_dir, testpath, fn)
            print(fd)                 #資料夾照片位置 
            fdd.append(str(fd))
            images.append(fd)
#呼叫函式，規範化圖片
pre_x = get_inputs(images)
#預測
pre_y = model.predict(pre_x)

j = 0
qq = 0
for i in range(len(pre_y)):
        #time.sleep(0.1)
        cc = fdd[i][34:]
        #j = j + 1 
        sickvalue =  1 - pre_y[i][0] #有病
        print("Time:"+str(datetime.datetime.now())[:19]+  "  病患ID:"+cc+"    發病機率為:  "+str(sickvalue*100)[:5]+"%")  
        main = " ID:"+cc+",發病率:"+str(sickvalue*100)[:5]+"%"
        payload=[{"id":"data", "value":[main]}]
        #response = requests.post(apiURL, headers=headers, data=json.dumps(payload))
        #print(fdd[i])  
        #print("測試資料數量:"+str(len(pre_y))+"誤判數量:"+str(j))
        #os.remove(fd)


        if(pre_y[i]> 0.9): #>0.9沒病
            j=j+1       
        else:
            qq=qq+1 
print(j)             
print(qq)    
  