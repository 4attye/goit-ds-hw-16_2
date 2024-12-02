import  streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import pickle

model_CNN = load_model('CNN.keras')
model_VGG16 = load_model('VGG16.keras')

models = ['CNN', 'VGG16']

st.title("Класифікація зображень за допомогою нейронної мережі")

uploaded_file = st.file_uploader("Завантажте зображення...", type=["jpg", "jpeg", "png"])

fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_history(hst):

    st.subheader('Графік втрат')
    plt.plot(hst['loss'], label='train')
    plt.plot(hst['val_loss'], label='test')
    plt.title('Loss')
    st.pyplot(plt)

    st.subheader('Графік точності')
    plt.plot([round(100*e, 2) for e in hst['acc']], label='train')
    plt.plot([round(100*e, 2) for e in hst['val_acc']], label='test')
    plt.title('Accuracy')
    st.pyplot(plt)

column = st.selectbox('Виберіть модель для аналізу:', models)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    if column == 'CNN':
        resized_image = image.resize((28, 28))
        resized_image = ImageOps.grayscale(resized_image)
        st.image(resized_image, caption='зображення 28x28', use_container_width=True)
        img_array = img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255
        preds = model_CNN.predict(img_array)
        predicted_class = np.argmax(preds[0])
        st.write(f"Предсказаний клас: {fashion_classes[predicted_class]}")
        st.write("Ймовірності для кожного класу:")
        for i, pred in enumerate(preds[0]):
            st.write(f"{fashion_classes[i]}: {pred:.4f}")

    elif column == 'VGG16':
        resized_image = image.resize((32, 32))
        st.image(resized_image, caption='зображення 32x32', use_container_width=True)
        img_array = img_to_array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255
        preds = model_VGG16.predict(img_array)
        predicted_class = np.argmax(preds[0])
        st.write(f"Предсказаний клас: {fashion_classes[predicted_class]}")
        st.write("Ймовірності для кожного класу:")
        for i, pred in enumerate(preds[0]):
            st.write(f"{fashion_classes[i]}: {pred:.4f}")

if column == 'CNN':
    with open('history_CNN.pkl', 'rb') as f:
        history = pickle.load(f)
    plot_history(history)

elif column == 'VGG16':
    with open('history_VGG16.pkl', 'rb') as f:
        history = pickle.load(f)
    plot_history(history)