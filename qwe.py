import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy
import time
import numpy as np
import pickle
import argparse
def create_resnet50_cifar10():
    input_tensor = Input(shape=(32, 32, 3))
    base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=input_tensor, pooling='avg')
    x = Flatten()(base_model.output)
    output_tensor = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

model = create_resnet50_cifar10()
total_params = model.count_params()
print(f'Total Parameters: {total_params}')