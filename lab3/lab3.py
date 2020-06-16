import cv2


face_csc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while(True):
    tf, img = cam.read()       
       
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_csc.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,0), 5)
        
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()