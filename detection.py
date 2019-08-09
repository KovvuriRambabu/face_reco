import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
path="dataSet"
id=0
def getProfile(id):
    conn=sqlite3.connect("faceBase.db")
    cmd="SELECT *FROM PERSON WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profeile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile


font = cv2.FONT_HERSHEY_SIMPLEX
bold=2
fontscale = 1
fontcolor = (255, 0, 0)
while True:
    ret, img =cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id, conf =rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        
        if profile!=None:
            
           
            cv2.putText(img,str(profile[1]),(x,y+h+30),font,fontscale,fontcolor,bold)
            cv2.putText(img,str(profile[2]),(x,y+h+60),font,fontscale,fontcolor,bold)
            cv2.putText(img,str(profile[3]),(x,y+h+90),font,fontscale,fontcolor,bold)
            #cv2.putText(img,str(profile[4]),(x,y+h+120),font,fontscale,fontcolor,bold)
    cv2.imshow('im',img) 
    if cv2.waitKey(10) ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
