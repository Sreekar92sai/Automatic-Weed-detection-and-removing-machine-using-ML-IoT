import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('weeds.h5')

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
