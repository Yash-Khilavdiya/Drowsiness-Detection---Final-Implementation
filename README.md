# Driver Drowsiness Detection System
Developed and Designed CNN, Har Cascade Library, Python, Flask.

It is a deep learning based project which detects or predicts the drowsiness state of a driver while driver using computer vision, cnn/ann, camera.

The approach for this project is to first train a model which detects the eyes of the driver and then predicts weather the eyes are open or closed of the driver. If the eyes of the driver is closed for more duration like 5-10 sec then an beep tone is generated which triggers the driver to wake up and it beeps until  the driver wakes up and when the user wakes up or eyes get opened then the beep sound gets turned off.
For predicting the state of eyes that they are closed or open a model is trained using a dataset which detects or predicts weather the eyes are open or closed and using that driver drowsy state is detected and alerts it. It is done using python programming language and deep learning concepts and libraries like keras for machine learning and deep learning. For Developing its web interface we have used flask framework. 
