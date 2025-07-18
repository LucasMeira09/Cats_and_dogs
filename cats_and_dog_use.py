import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

img_path = 'code/2.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))

img_array = tf.keras.preprocessing.image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

model = load_model("code/modelo_gatos_cachorros.h5")

prediction = model.predict(img_array)

classe= 'Dog' if prediction[0] > 0.5 else 'Cat'
print(f"Prevision: {classe} ({prediction[0][0]:.2f})")

matplotlib.use('TkAgg')

plt.imshow(img)
plt.title(classe)
plt.axis('off')
plt.show()