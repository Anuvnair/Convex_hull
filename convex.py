import cv2
import numpy as np
import math
#cap = cv2.VideoCapture(0)
#while(cap.isOpened()):
# ret, img = cap.read()
img = cv2.imread('F1.png')
#cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
#crop_img = img[100:300, 100:300]

lower = np.array([0,90,100])
upper = np.array([20,150,250])

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dil = cv2.dilate(mask, kernel, iterations=1)
#grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#value = (35, 35)
#blurred = cv2.GaussianBlur(grey, value, 0)
#ret,thresh1 = cv2.threshold(blurred, 127, 255,
                               #cv2.THRESH_BINARY_INV)
#t2=thresh1.copy()
#cv2.imshow('Thresholded', thresh1)

image, contours, hierarchy = cv2.findContours(dil,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


cnt = max(contours, key = lambda x: cv2.contourArea(x))
    
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),0)
hull = cv2.convexHull(cnt)
drawing = np.zeros(img.shape,np.uint8)
#cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
#cv2.drawContours(drawing,[hull],0,(0,0,255),0)
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)
count_defects = 0
#cv2.drawContours(img, contours, -1, (0,255,0), 3)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)#green line
    cv2.circle(img,far,5,[0,0,255],-1)#red circle
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
    if angle <= 90:
        count_defects += 1
        cv2.circle(img,far,1,[0,0,255],-1)
        #dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(img,start,end,[0,255,0],2)
        #cv2.circle(crop_img,far,5,[0,0,255],-1)
if count_defects == 1:
    cv2.putText(img,"1", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
elif count_defects == 2:
    str = "2"
    cv2.putText(img, str, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
elif count_defects == 3:
    cv2.putText(img,"3", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
elif count_defects == 4:
    cv2.putText(img,"4", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
else:
    cv2.putText(img,"5", (50,50),\
                cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
#cv2.imshow('drawing', drawing)
#cv2.imshow('end', crop_img)
cv2.imshow('Gesture', img)
#all_img = np.hstack((drawing, img))
#cv2.imshow('Contours', all_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
