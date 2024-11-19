import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Set image dimensions
IMG_SIZE = 150
BATCH_SIZE = 32

# Initialize ImageDataGenerators for training and validation/test
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    zoom_range=0.2, 
    shear_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
# When using ImageDataGenerator with flow_from_directory(),
# the function expects that your data is organized into separate folders for each class. i.e test/normal, test/pneumonia
train_generator = train_datagen.flow_from_directory(
    'dataset/train', 
    target_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    'dataset/val', 
    target_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    'dataset/test', 
    target_size=(IMG_SIZE, IMG_SIZE), 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

sample_batch = next(train_generator)
sample_images, sample_labels = sample_batch

print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", val_generator.samples)

# Plot a few augmented images
# images in the normal folder are labeled 0, and those in the pneumonia folder are labeled 1
plt.figure(figsize=(10, 10))
for i in range(9):  # Displaying 9 images
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f'Label: {int(sample_labels[i])}')
    plt.axis('off')

plt.show()

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=10  # Adjust this based on your needs

    
)


# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f'Test accuracy: {test_acc}')

# Save the trained model
model.save('pneumonia_detector_model.keras')
