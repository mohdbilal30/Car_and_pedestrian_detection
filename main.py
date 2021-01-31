import cv2
import numpy as np
import matplotlib.pyplot as plt

### importing video files ###

### capturing Video
vid1 = cv2.VideoCapture("busy_street.mp4") # add a video file 
vid2 = cv2.VideoCapture("car_dash_cam.mp4") # add a video file 
vid3 = cv2.VideoCapture("cars_dash_cam2.mp4") # add a video file

##### Defining xml files for features to detect the face #####
##### Machine Learning Files (Haar Cascade xml files) for pedestrials #####
pedestrials_cascade = cv2.CascadeClassifier("pedestrians.xml")
##### Machine Learning Files (Haar Cascade xml files) for cars #####
car_cascade = cv2.CascadeClassifier("cars.xml")

##### Function to display images in matplotlib with color correction ####

def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
#### Converting image to gray scale ####
    rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_image)
    plt.show()

#### Function to detect cars and Pedestrials ####

def detect(img):

    img_copy = img.copy()

#### Detection padestrials using detectMultiScale to get rectangle ####
    pede_rects = pedestrials_cascade.detectMultiScale(img_copy)

#### Detection padestrials using detectMultiScale to get rectangle ####

    cars_rects = car_cascade.detectMultiScale(img_copy,scaleFactor=1.1,minNeighbors=2)

#### for loop to draw rectangle on detected pedestrials ####

    for (x,y,w,h) in pede_rects:

#### drawing rectangle ####
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),3)

#### for loop to draw rectangle on detected cars ####
    for (x,y,w,h) in cars_rects:

        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,255),3)

    return img_copy
##### While loop for capturing continuous frames #####
while True:

    ret,frame = vid1.read()
##### safe coding approch #####
    if ret == True:
        frame = detect(frame)

        new_frame = cv2.resize(frame,dsize=(800,700))
##### function to show captured frame #####
        cv2.imshow("Cars and Pedestrials Detectin AI",new_frame)
    else:
##### breaking out of loop if there is no capture #####
        break

    k = cv2.waitKey(1) & 0xFF
##### press q on keyboard to quit AI system
    if k == ord("q"):
##### break the while True loop to close the AI system
        break
##### releasing all capture  and windows #####
vid3.release()
cv2.destroyAllWindows()
