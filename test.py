"""
Created on Wed May 13 12:09:24 2020
@author: Kraken

Project: Prediction using trained generator models
"""
# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

if not os.path.exists('prediction_results'):
    os.mkdir('prediction_results')

INPUT_DIM = 100
SAMPLES = 20


# Load models
def get_model(model_no=None):
    model_list = []
    for file_name in os.listdir('model_evaluation'):
        if file_name.startswith('g_model'):
            model_list.append(file_name)
    model = load_model('model_evaluation/' + model_list[-1])
    return model, model_list


def create_input(input_dim, samples):
    # randn function creates values between [0, 1]
    x_test = np.random.randn(input_dim, samples)
    x_test = x_test.reshape(samples, input_dim)
    return x_test


def save_images(model_pred, samples):
    for i in range(samples):
        fig = plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.imshow(model_pred[i][:, :, -1] * 255, cmap='gray_r')
        plt.savefig('prediction_results/img_' + str(i + 1) + '.png')
        plt.close()
    fig = plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(model_pred[i][:, :, -1] * 255, cmap='gray_r')
    plt.savefig('summary.png', dpi=200)
        


if __name__ == "__main__":
    model, model_list = get_model()
    X = model.predict(create_input(INPUT_DIM, SAMPLES))
    save_images(X, SAMPLES)
