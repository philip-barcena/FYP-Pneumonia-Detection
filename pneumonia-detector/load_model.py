import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model = load_model('pneumonia_detector_model.h5')

print("Model input shape:", model.input_shape)

# Check the model's input shape
print("Model input shape:", model.input_shape)

# Prepare your input image
def prepare_image(image_path):
    # Load the image and resize it to the expected input size
    img = load_img(image_path, target_size=(150, 150))  # Adjust based on your model's expected size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Scale pixel values to [0, 1]
    return img_array

# Predict pneumonia
def predict(image_path):
    image = prepare_image(image_path)
    prediction = model.predict(image)
    return prediction

# Example usage
result = predict('testimage.jpeg')  # Replace with your image path
print("Prediction:", result)
