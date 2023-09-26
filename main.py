
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, Lambda, Flatten, Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from network import create_base_network_signet
from utils import euclidean_distance, eucl_dist_output_shape


img_w, img_h = 150, 220
input_shape=(img_h, img_w, 1)



base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

processed_a = base_network(input_a)
processed_b = base_network(input_b)



distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

outputs = Dense(1, activation="sigmoid")(distance)

model = Model(inputs=[input_a, input_b], outputs=outputs)

model.load_weights('C:/Users/osy/Desktop/signrecwsiamese/signet.h5')

threshold = 0.5

img1 = cv2.imread("C:/Users/osy/Desktop/signrecwSiamese/r-sign.jpg")
img2 = cv2.imread("C:/Users/osy/Desktop/signrecwSiamese/f-sign.jpg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1 = cv2.medianBlur(img1, 5)
img2 = cv2.medianBlur(img2, 5)

img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

img1 = cv2.resize(img1, (img_w, img_h))
img2 = cv2.resize(img2, (img_w, img_h))


img1 = np.array(img1, dtype = np.float64)
img2 = np.array(img2, dtype = np.float64)
img1 /= 255
img2 /= 255
img1 = img1[..., np.newaxis]
img2 = img2[..., np.newaxis]


pred, tr_y = [], []

model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])[0][0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
ax1.imshow(np.squeeze(img1), cmap='gray')
ax2.imshow(np.squeeze(img2), cmap='gray')
ax1.set_title('Gercek Imza')

ax1.axis('off')
ax2.axis('off')

result = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
diff = result[0][0]
print("Difference Score = ", diff)
if diff > threshold:
    print("Sahte Imza")
    ax2.set_title('Sahte Imza')
else:
    print("Gercek Imza")
    ax2.set_title('Gercek Imza')

plt.show()
