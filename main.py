import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.yolov4 import YOLOv4

# Load the YOLOv4 model
yolov4 = YOLOv4(weights='yolov4.weights', input_shape=(None, None, 3))

# Function to make predictions
def predict(image):
    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0)
    image_np = image_np / 255.0  # Normalize image

    # Make prediction using YOLOv4 model
    boxes, scores, classes, nums = yolov4.predict(image_np)

    return boxes, scores, classes, nums

def main():
    st.title('YOLOv4 Object Detection')
    st.markdown('Upload an image for object detection.')

    uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Objects'):
            boxes, scores, classes, nums = predict(image)

            st.markdown('### Detected Objects:')
            for i in range(nums[0]):
                class_name = yolov4.class_names[int(classes[0][i])]
                score = float(scores[0][i])
                st.write(f'- {class_name}: {score:.4f}')

if __name__ == '__main__':
    main()
