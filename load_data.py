import tensorflow as tf

dataset = tf.keras.datasets.cifar10.load_data()
import os
import numpy as np
import tensorflow as tf

# Cargue el conjunto de datos cifar10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Cree una carpeta para almacenar los datos
folder_path = "images_streamlit"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Guarde las imágenes y etiquetas como archivos .npy
np.save(os.path.join(folder_path, "x_train.npy"), x_train)
np.save(os.path.join(folder_path, "y_train.npy"), y_train)
np.save(os.path.join(folder_path, "x_test.npy"), x_test)
np.save(os.path.join(folder_path, "y_test.npy"), y_test)
