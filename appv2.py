from flask import Flask, render_template, Response, request
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import io
from PIL import Image

app = Flask(__name__, template_folder='Templates', static_folder='Static')

mixer.init()
driver_sound = mixer.Sound('driver_alert.wav')
front_pass_sound = mixer.Sound('Front_Passenger_alert.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

model = load_model('models/Open_Closed_Eyes_Detection.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
driver_score=0
front_passenger_score=0

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_frames():  # Generate camera frames
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    count=0
    driver_score=0
    front_passenger_score=0
    thicc=2
        
    while(True):
        driver_rpred_max_index = -1
        driver_lpred_max_index = -1
        front_pass_rpred_max_index = -1
        front_pass_lpred_max_index = -1
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2] 
        cv2.rectangle(frame, (319,0) , (321,height) , (255,255,255) , thickness=cv2.FILLED)
    #     print(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     print(gray.shape)
        
        faces = face.detectMultiScale(gray,minNeighbors=5, scaleFactor=1.1, minSize=(25,25))

        # cv2.putText(frame,"Side Passenger",(0,10), font, 1, (255,255,255), 1, cv2.LINE_AA)
        # cv2.putText(frame,"Driver", (320,10), font, 1, (255,255,255), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED)
        cv2.rectangle(frame, (320,height-50) , (200+320,height) , (0,0,0) , thickness=cv2.FILLED )
        cv2.rectangle(frame, (0,0) , (200,50) , (0,0,0) , thickness=cv2.FILLED)
        cv2.rectangle(frame, (320,0) , (320+200,50) , (0,0,0) , thickness=cv2.FILLED)
        # cv2.rectangle(frame, (320,400) , (200+320,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)
        

        driver_image  = gray[:, 320:]
        cv2.imshow('Driver', driver_image)

        front_passenger = gray[:, :320]
        cv2.imshow('Front Passenger', front_passenger)
            
        driver_left_eye = leye.detectMultiScale(driver_image)
        driver_right_eye =  reye.detectMultiScale(driver_image)
        
        front_pass_left_eye = leye.detectMultiScale(front_passenger)
        front_pass_right_eye =  reye.detectMultiScale(front_passenger)
        
        # Predicting weather eyes of driver are closed or open
        for (x,y,w,h) in driver_right_eye:
            cv2.rectangle(frame, (x+320,y) , (x+w+320,y+h) , (100,100,100) , 1 )
            driver_right_eye=driver_image[y:y+h,x:x+w]
            count=count+1
    #         driver_right_eye = cv2.cvtColor(driver_right_eye,cv2.COLOR_BGR2GRAY)
            driver_right_eye = cv2.resize(driver_right_eye,(24,24))
            driver_right_eye= driver_right_eye/255
            driver_right_eye=  driver_right_eye.reshape(24,24,-1)
            driver_right_eye = np.expand_dims(driver_right_eye,axis=0)
            driver_rpred = model.predict(driver_right_eye)
            driver_rpred_max_index = driver_rpred.argmax()

        for (x,y,w,h) in driver_left_eye:
            cv2.rectangle(frame, (x+320,y) , (x+w+320,y+h) , (100,100,100) , 1 )
            driver_left_eye=driver_image[y:y+h,x:x+w]
            count=count+1
    #         driver_left_eye = cv2.cvtColor(driver_left_eye,cv2.COLOR_BGR2GRAY)  
            driver_left_eye = cv2.resize(driver_left_eye,(24,24))
            driver_left_eye= driver_left_eye/255
            driver_left_eye=driver_left_eye.reshape(24,24,-1)
            driver_left_eye = np.expand_dims(driver_left_eye,axis=0)
            driver_lpred = model.predict(driver_left_eye)
            driver_lpred_max_index = driver_lpred.argmax()
            
        for (x,y,w,h) in front_pass_right_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            front_pass_right_eye=front_passenger[y:y+h,x:x+w]
            count=count+1
    #         front_pass_right_eye = cv2.cvtColor(front_pass_right_eye,cv2.COLOR_BGR2GRAY)
            front_pass_right_eye = cv2.resize(front_pass_right_eye,(24,24))
            front_pass_right_eye= front_pass_right_eye/255
            front_pass_right_eye=  front_pass_right_eye.reshape(24,24,-1)
            front_pass_right_eye = np.expand_dims(front_pass_right_eye,axis=0)
            front_pass_rpred = model.predict(front_pass_right_eye)
            front_pass_rpred_max_index = front_pass_rpred.argmax()

        for (x,y,w,h) in front_pass_left_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
            front_pass_left_eye=front_passenger[y:y+h,x:x+w]
            count=count+1
    #         front_pass_left_eye = cv2.cvtColor(front_pass_left_eye,cv2.COLOR_BGR2GRAY)  
            front_pass_left_eye = cv2.resize(front_pass_left_eye,(24,24))
            front_pass_left_eye= front_pass_left_eye/255
            front_pass_left_eye=front_pass_left_eye.reshape(24,24,-1)
            front_pass_left_eye = np.expand_dims(front_pass_left_eye,axis=0)
            front_pass_lpred = model.predict(front_pass_left_eye)
            front_pass_lpred_max_index = front_pass_lpred.argmax()

        
        if(driver_lpred_max_index==0 and driver_rpred_max_index==0):
            driver_score=driver_score+1
            cv2.putText(frame,"Closed",(10+320,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,'Score:'+str(driver_score),(100+320,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    #    if(rpred[0]==1 or lpred[0]==1):
        else:
            driver_score=driver_score-1
            cv2.putText(frame,"Open",(10+320,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,'Score:'+str(driver_score),(100+320,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
        if(driver_score<0):
            driver_score=0   
            cv2.putText(frame,'Score:'+str(driver_score),(100+320,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
        if(front_pass_lpred_max_index==0 and front_pass_rpred_max_index==0):
            front_passenger_score=front_passenger_score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(frame,'Score:'+str(front_passenger_score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    #    if(rpred[0]==1 or lpred[0]==1):
        else:
            front_passenger_score=front_passenger_score-1
            cv2.putText(frame,"Open", (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame,'Score:'+str(front_passenger_score), (100,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
            
        if(front_passenger_score<0):
            front_passenger_score=0   
            cv2.putText(frame,'Score:'+str(front_passenger_score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)


        print('Driver Score : ', driver_score)
        print('Front Passenger Score: ', front_passenger_score)
        
        # Code to beep alarm when driver score is greater than 10
        if(driver_score>5):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'driver_image.jpg'),frame)
            try:
                driver_sound.play()
            except:  # isplaying = False
                pass

            if(thicc<8):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(320,0),(width,height),(0,0,255),thicc) 
        
        elif (driver_score < 5):
            driver_sound.stop()
        
        # Code to beep alarm when Front Passenger score is greater than 10
        if(front_passenger_score>5):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path,'front_passenger.jpg'),frame)
            try:
                front_pass_sound.play()
            except:  # isplaying = False
                pass

            if(thicc<8):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(320,height),(0,0,255),thicc) 
        
        elif (front_passenger_score < 5):
            front_pass_sound.stop()

        # Save the image
        cv2.imwrite("frame_img.jpg", frame)

        # Load saved image
        frame_img = Image.open('frame_img.jpg')

        # Convert the above loaded image to bytes
        buffer = io.BytesIO()
        frame_img.save(buffer, format='JPEG')
        frame2 = buffer.getvalue()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
        
    # cap.release()



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)