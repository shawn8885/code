import cv2
from imutils import paths
import os 

#Train resize
image_train = list(paths.list_images('D:/shawn/new/ph2'))

for i in range(len(image_train)):
    image = cv2.imread(image_train[i])
    path,name = os.path.split(image_train[i])
    name = name.split(image_train[i])
    image = cv2.resize(image,(140,140))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(path,'{}.jpg'.format(name)),image)
    #cv2.imwrite('C:/Users/User/Desktop/New_Rice_Dataset/train',image)
print("ok")