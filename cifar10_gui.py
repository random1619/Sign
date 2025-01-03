import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas, PhotoImage
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import cv2

# Load the trained model
MODEL_PATH = "cifar10_model.h5"  # Ensure the model path is correct
model = load_model(MODEL_PATH)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Function to classify the uploaded image
def classify_image():
    global uploaded_image_path

    if not uploaded_image_path:
        result_label.config(text="Please upload an image first!")
        return

    # Preprocess and predict
    image = preprocess_image(uploaded_image_path)
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    result_label.config(text=f"Predicted Class: {class_names[class_idx]}")

# Function to upload an image
def upload_image():
    global uploaded_image_path
    uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if uploaded_image_path:
        # Display uploaded image
        image = Image.open(uploaded_image_path)
        image = image.resize((150, 150))
        img = ImageTk.PhotoImage(image)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="Image uploaded successfully! Ready to classify.")

# Initialize Tkinter window
window = tk.Tk()
window.title("CIFAR-10 Image Classifier")
window.geometry("400x500")

uploaded_image_path = None

# GUI components
upload_btn = Button(window, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)

image_label = Label(window)
image_label.pack(pady=10)

classify_btn = Button(window, text="Classify Image", command=classify_image)
classify_btn.pack(pady=10)

result_label = Label(window, text="", font=("Arial", 14))
result_label.pack(pady=20)

# Run the Tkinter event loop
window.mainloop()
