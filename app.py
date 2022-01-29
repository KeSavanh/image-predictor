
from flask import Flask, render_template, request

import matplotlib.pyplot as plt



import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
  return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
  imagefile = request.files['imagefile']
  image_path = './Images/' + imagefile.filename
  imagefile.save(image_path)

  image_size = (299, 299)
  img = image.load_img(image_path, target_size=image_size)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  model = Xception(include_top=True, weights="imagenet")
  predictions = model.predict(x)
  label = decode_predictions(predictions)
  label = label[0][0]

  classification = '%s (%.2f%%' % (label[1], label[2]*100)
  

  return render_template('index.html', prediction=classification)


if __name__ == '__main__':
  app.run(port=3000, debug=True)