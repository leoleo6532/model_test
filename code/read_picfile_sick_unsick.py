#導入
import os
import shutil
#定義
testpic = []
testtxt = []
testtxtgg = []


dir1='C:\\lasttest\\DL\\LungCancerDetection-master\\LungCancerDetection-master\\src\\data\\val'

#執行
for filename in os.listdir(dir1):
    testpic.append(filename)


with open("C:\\lasttest\\DL\\LungCancerDetection-master\\LungCancerDetection-master\\src\\data\\valdatalabels.txt", 'r') as fp:
    for line in fp.readlines():
       testtxt.append(line[-2])
       testtxtgg.append(line[4:-2])
print(testtxtgg)



for i in range(len(testpic)):
       if (str(testtxt[i]) == "0"): 
         shutil.copyfile(dir1+'\\'+str(testtxtgg[i]),"C:\\lasttest\\DL\\leo\\val\\unsick\\"+str(testtxtgg[i]))  
         print(testtxtgg[i])
          