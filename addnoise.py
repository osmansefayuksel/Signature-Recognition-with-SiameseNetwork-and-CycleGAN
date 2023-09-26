import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def add_noise(image):

    height, width = image.shape[:2]
    num_lines = np.random.randint(1, 5) 
    y0 = int(height/num_lines) 
    for i in range(num_lines):
        line_thickness = np.random.randint(1, 5)
        x1, x2 = 0, width 
        y = y0*(i+1) + np.random.randint(-0.05*height, 0.05*height) 
        image = cv2.line(image, (x1, y), (x2, y), (0, 0, 0), thickness=line_thickness) 
        prev_y = y

    return image


def add_name(image):

    height, width = image.shape[:2]
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_DUPLEX]
    texts = ['Osman Sefa Yuksel', 'Berkay Kilic', 'Hatice Betul Ozer', 'Ersin Kilic', 'Omer Faruk Yildirim', 'Hakki Mert Peyk', 'Busra Karabulut']
    font_scale = np.random.random() + 1.3
    thickness = np.random.randint(1, 3)
    y = np.random.randint(0.75*height, 1.02*height)
    x = np.random.randint(0.0005*width, 0.3*width)
    font = np.random.choice(fonts)
    text = np.random.choice(texts)
    font_color = (0, 0, 0)
    image = cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    return image


