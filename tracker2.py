import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

def detect_dab(arms, head):
    if len(arms)<2:
        print("1")
        return False
    for i in range(2):
       print("4")
       if arms[i][2] < 2*arms[i][3] :
         print("2")
         return False
       if(head[1] - arms[i][1] > head[3]*1.5):
         print("3")
         return False
    print("5")
    return True

while(1):

    try:
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)

        roi=frame[0:900, 0:900]

        cv2.rectangle(frame,(0,0),(900,900),(0,255,0),0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        mask = cv2.erode(mask,kernel,iterations = 4)

        mask = cv2.GaussianBlur(mask,(5,5),200)


        ret,thresh = cv2.threshold(mask, 70, 255, 0)
        _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(roi, contours, -1, 255, 3)
        body_parts = []
        arms = []
        head = []
        for _ in range(3):
            cnt = max(contours, key = cv2.contourArea)

            if cv2.contourArea(cnt) > 1000:

                epsilon = 0.0005*cv2.arcLength(cnt,True)
                approx= cv2.approxPolyDP(cnt,epsilon,True)
                hull = cv2.convexHull(cnt)
                areacnt = cv2.contourArea(cnt)
                hull = cv2.convexHull(approx, returnPoints=False)
                defects = cv2.convexityDefects(approx, hull)
                x,y,w,h = cv2.boundingRect(cnt)
                body_parts.append([x,y,w,h])

                cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)

                #(x,y),radius = cv2.minEnclosingCircle(cnt)
                #center = (int(x),int(y))
                #radius = int(radius)
                #cv2.circle(roi,center,radius,(0,255,255),2)

                #font = cv2.FONT_HERSHEY_SIMPLEX
                #if areacnt<2000:
                #    cv2.putText(frame,'...?',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

            contours.remove(cnt)

        for num in range(len(body_parts)-1,0,-1):
                for i in range(num):
                    if body_parts[i][0]>body_parts[i+1][0]:
                        temp = body_parts[i]
                        body_parts[i] = body_parts[i+1]
                        body_parts[i+1] = temp
        print(body_parts)
        arms = [body_parts[0],body_parts[2]]
        head = body_parts[1]
        #check for dab
        if detect_dab(arms, head):
            print("detected dab")
            cv2.putText(frame,'dab!',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except Exception as e: print(e)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
    



