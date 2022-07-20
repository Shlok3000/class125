import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps
import os, ssl

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0
clf  = LogisticRegression(solver = "saga", multi_class="multinomial").fit(x_train_scaled, y_train)

def getPrediction(image):
    im_pil = Image.open(image)
    ibw = im_pil.convert("L")
    img_bw_resized = ibw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resized,pixel_filter)
    #using clip to limit the values betwn 0-255
    img_inverted_scale = np.clip(img_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resized)
    img_inverted_scale = np.asarray(img_inverted_scale)/max_pixel
    #converting into an array() to be used in model for prediction
    test_sample = np.array(img_inverted_scale).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return(test_pred[0])
    