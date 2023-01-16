import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
#from PIL import ImageGrab


path = 'ImagesBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
# Enlever le .png
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
# STEP 2 : Find the encoders
def findEncondings(images):
    encodeList = []
    for img in images:
        #Transformation des image en RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #Find the encoding
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
#def captureScreen(bbox=(300,300,690+300,530+300)):
   # capScr = np.array(ImageGrab.grab(bbox))
   # capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    #return capScr

encodeListKnown = findEncondings(images)
print('Encoding Complete ')

#STEP3 : find the matches between our encoders

#Initialize the camera

cap = cv2.VideoCapture(0)  #0 IS THE ID
while True:
    success, img=cap.read()
    # we do it in the reel time so we reduce the size of the images
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)  # 1/4 of the size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # To find the location of our images and find the encoders to our web camera
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurrFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0.255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0.255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)





