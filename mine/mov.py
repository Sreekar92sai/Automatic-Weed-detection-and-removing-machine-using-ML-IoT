import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from tensorflow.keras.models import load_model

# Load weed detection model
weed_model = load_model('weed_detection_model.h5')

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize GPIO pins for wheel movement
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

# Initialize GPIO pins for blade cutting
GPIO.setup(26, GPIO.OUT)
GPIO.setup(32, GPIO.OUT)

# Initialize GPIO pins for physical markers
GPIO.setup(36, GPIO.IN)
GPIO.setup(38, GPIO.IN)

# Define a function for physical marker detection
def marker_detection():
    # Check if the left marker is detected
    if GPIO.input(36):
        return 'left'
    # Check if the right marker is detected
    elif GPIO.input(38):
        return 'right'
    # No marker detected
    else:
        return None

# Define a function for obstacle detection
def obstacle_detection():
    # Measure the distance using the ultrasonic sensor
    GPIO.output(12, True)
    time.sleep(0.00001)
    GPIO.output(12, False)
    while GPIO.input(18) == 0:
        pulse_start = time.time()
    while GPIO.input(18) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return distance

# Loop through each frame from the camera
while True:
    # Capture the frame from the camera
    ret, frame = camera.read()
    
    # Detect physical markers
    marker = marker_detection()
    
    if marker == 'left':
        # Turn left
        GPIO.output(16, False)
        GPIO.output(18, True)
        GPIO.output(22, True)
        GPIO.output(24, False)
        
    elif marker == 'right':
        # Turn right
        GPIO.output(16, True)
        GPIO.output(18, False)
        GPIO.output(22, False)
        GPIO.output(24, True)
        
    else:
        # Move forward
        GPIO.output(16, True)
        GPIO.output(18, False)
        GPIO.output(22, True)
        GPIO.output(24, False)
        
        # Display the frame
        cv2.imshow('frame', frame)
        
        # Check for weeds and activate the blade cutter if needed
        # Resize the frame to the input size of the model
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        
        # Preprocess the image
        img = img / 255.0
        
        # Predict the class of the image
        pred = weed_model.predict(img)
        pred_class = np.argmax(pred, axis=1)
        
        # Check for obstacle
        distance = obstacle_detection()
        if distance < 20:
            # Stop the wheel movement and wait for obstacle to clear
            GPIO.output(16, False)
            GPIO.output(18, False)
            GPIO.output(22, False)
           
    def obstacle_detection():
        # Send a trigger signal to the ultrasonic sensor
        GPIO.output(11, True)
        time.sleep(0.00001)
        GPIO.output(11, False)

        # Measure the duration of the echo signal
        start_time = time.time()
        stop_time = time.time()
        while GPIO.input(13) == 0:
            start_time = time.time()
        while GPIO.input(13) == 1:
            stop_time = time.time()

        # Calculate the distance to the obstacle
        elapsed_time = stop_time - start_time
        distance = (elapsed_time * 34300) / 2

        return distance


while True:
    # Capture the frame from the camera
    ret, frame = camera.read()
    # Detect physical markers
marker = marker_detection()

if marker == 'left':
    # Turn left
    GPIO.output(16, False)
    GPIO.output(18, True)
    GPIO.output(22, True)
    GPIO.output(24, False)
    
elif marker == 'right':
    # Turn right
    GPIO.output(16, True)
    GPIO.output(18, False)
    GPIO.output(22, False)
    GPIO.output(24, True)
    
else:
    # Move forward
    GPIO.output(16, True)
    GPIO.output(18, False)
    GPIO.output(22, True)
    GPIO.output(24, False)
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Check for obstacles and activate the obstacle avoidance mechanism if needed
    distance = obstacle_detection()
    if distance < 20:
        # Stop the wheel movement and wait for obstacle to clear
        GPIO.output(16, False)
        GPIO.output(18, False)
        GPIO.output(22, False)
        GPIO.output(24, False)
        time.sleep(1)
    else:
        # Move the wheels forward
        GPIO.output(16, True)
        GPIO.output(18, False)
        GPIO.output(22, True)
        GPIO.output(24, False)
    
        # Weed detection and blade cutting
        if pred_class == 1:
            # Turn on the blade cutting motor and wait for 1 second
            GPIO.output(26, True)
            GPIO.output(32, False)
            time.sleep(1)
            GPIO.output(26, False)
        else:
            # Turn off the blade cutting motor
            GPIO.output(26, False)
    
    # Show the frame
    cv2.imshow('frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

GPIO.cleanup()
