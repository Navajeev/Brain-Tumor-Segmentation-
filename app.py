import streamlit as st
import cv2
import numpy as np
from segmentation import segment_tumor

st.title("Brain Tumor Segmentation")
st.write("Upload a brain MRI image to detect and segment tumor region.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    segmented = segment_tumor(image)

    st.image(segmented, caption="Segmented Tumor", use_column_width=True)
    st.success("Segmentation complete!")
