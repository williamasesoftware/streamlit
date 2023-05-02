import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Crear la aplicación
st.title("Visualizador de imágenes CIFAR-10")

# Obtener la ruta de la carpeta que contiene los archivos .npy
folder_path = st.text_input("Ingrese la ruta de la carpeta que contiene los archivos .npy")

# Cargar los datos desde los archivos .npy
if folder_path:
    x_train = np.load(os.path.join(folder_path, "x_train.npy"))
    y_train = np.load(os.path.join(folder_path, "y_train.npy"))
    x_test = np.load(os.path.join(folder_path, "x_test.npy"))
    y_test = np.load(os.path.join(folder_path, "y_test.npy"))

    # Obtener 30 imágenes aleatorias y sus etiquetas
    random_indices = np.random.randint(0, len(x_train), size=30)
    images = x_train[random_indices]
    labels = y_train[random_indices]

    # Crear una figura con subplots
    fig, axes = plt.subplots(6, 5, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    # Definir el diccionario para mapear los valores enteros de la etiqueta a su correspondiente texto
    label_names = {
        0: "Avión",
        1: "Automóvil",
        2: "Pájaro",
        3: "Gato",
        4: "Ciervo",
        5: "Perro",
        6: "Rana",
        7: "Caballo",
        8: "Barco",
        9: "Camión"
    }

    # Mostrar cada imagen en un subplot con su etiqueta
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"{label_names[labels[i][0]]}")

    # Mostrar la figura en Streamlit
    st.pyplot(fig)

    # Obtener una imagen aleatoria y su etiqueta
    random_index = np.random.randint(0, len(x_train))
    image = x_train[random_index]
    label = y_train[random_index][0]

    # Crear una figura en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Obtener los valores de R, G y B de la imagen y convertirlos a un rango de 0 a 1
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Mostrar los puntos en el espacio RGB como una gráfica de dispersión
    ax.scatter(r.flatten(), g.flatten(), b.flatten(), c=image.reshape(-1, 3) / 255)

    # Configurar los ejes y la etiqueta de título
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    ax.set_title(f"Imagen #{random_index} es {label_names[label]}")

    # Añadir un slider para rotar la gráfica en el eje x
    if "angle" not in st.session_state:
        st.session_state.angle = 0
    angle = st.slider("Ángulo de rotación (grados)", 0, 360, st.session_state.angle)

    ax.view_init(elev=30., azim=angle)

    # Mostrar la figura en Streamlit
    st.pyplot(fig)

    

