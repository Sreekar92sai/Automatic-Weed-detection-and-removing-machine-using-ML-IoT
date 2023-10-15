import cv2
import numpy as np
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
import time
from threading import Thread

# Load the trained model
model = load_model('weeds.h5')

# Set up GPIO for LED control
GPIO.setmode(GPIO.BOARD)
GPIO.setup(38, GPIO.OUT)

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# Define a function to detect weed in a separate thread
def detect_weed():
    global camera, model
    
    while True:
        # Capture the frame from the camera
        ret, frame = camera.read()

        # Resize the frame to the input size of the model
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)

        # Preprocess the image
        img = img / 255.0

        # Predict the class of the image
        pred = model.predict(img)
        pred_class = np.argmax(pred, axis=1)

        # Control the LED based on the prediction
        if pred_class == 1:
            GPIO.output(38, GPIO.HIGH) # Turn on LED
        else:
            GPIO.output(38, GPIO.LOW) # Turn off LED

# Start the weed detection thread
t = Thread(target=detect_weed)
t.daemon = True
t.start()

# Loop through each frame from the camera
while True:
    # Capture the frame from the camera
    ret, frame = camera.read()
    
    # Display the result on the frame
    if pred_class == 1:
        cv2.putText(frame, 'Weed', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No weed', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # Show the frame
    cv2.imshow('frame', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
GPIO.cleanup()
