# Import necessary libraries
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.applications import VGG16

# Define the input size of the images
IMAGE_SIZE = [224, 224]

# Load the VGG16 model
vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the layers of the VGG16 model
for layer in vgg16.layers:
    layer.trainable = False

# Get the number of output classes
folders = glob('Dataset/Train/*')
num_classes = len(folders)

# Add custom layers to the VGG16 model
x = Flatten()(vgg16.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Create a new model using the modified VGG16 model and the custom layers
model = Model(inputs=vgg16.input, outputs=x)

# Define the hyperparameters for Bayesian optimization
tuner = BayesianOptimization(
    model,
    objective='val_accuracy',
    max_trials=5,
    directory='bayesian_opt',
    project_name='weed_detection'
)

# Define the training and validation data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    'Dataset/Train',
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'Dataset/Test',
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Train the model using Bayesian optimization
tuner.search(train_set, epochs=5, validation_data=test_set)

# Print the summary of the best model
tuner.results_summary()

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Compile the best model
best_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.0001),
    metrics=['accuracy']
)

# Train the best model on the training and validation data
best_model.fit(
    train_set,
    validation_data=test_set,
    epochs=5,
    batch_size=32,
    steps_per_epoch=len(train_set),
    validation_steps=len(test_set)
)

# Save the best model
best_model.save('weeds.h5')
