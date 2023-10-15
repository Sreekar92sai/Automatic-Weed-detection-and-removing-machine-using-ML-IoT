import cv2
import numpy as np
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO
import time

# Load the trained model
model = load_model('weeds.h5')

# Set up GPIO for LED control
GPIO.setmode(GPIO.BOARD)
GPIO.setup(38, GPIO.OUT)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Loop through each frame from the camera
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
    if pred[0][pred_class] > 0.5:
        # If the prediction probability is greater than 0.5, it's weed
        print("Weed detected")
        GPIO.output(38, GPIO.HIGH) # Turn on LED
    else:
        # Otherwise, it's not weed
        print("No weed detected")
        GPIO.output(38, GPIO.LOW) # Turn off LED
        
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
