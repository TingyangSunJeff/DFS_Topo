#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Creates a ResNet50 model for CIFAR-10 using pretrained weights.
    The model resizes CIFAR-10 images to 224x224 so that the network architecture,
    originally designed for ImageNet, can be fully leveraged.
    """
    # Define input tensor
    input_tensor = Input(shape=input_shape)
    # Load the ResNet50 model pretrained on ImageNet without its top layers
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    # Add a global average pooling layer, batch normalization, dropout, and final dense layer
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

def main():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Upscale images from 32x32 to 224x224 for ResNet50.
    x_train = tf.image.resize(x_train, (224, 224)).numpy()
    x_test = tf.image.resize(x_test, (224, 224)).numpy()
    
    # Preprocess images using ResNet50 preprocessing
    x_train = preprocess_input(x_train.astype('float32'))
    x_test = preprocess_input(x_test.astype('float32'))
    
    # One-hot encode labels.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Create the model.
    model = create_model(input_shape=(224, 224, 3), num_classes=10)
    
    # Freeze the base model initially.
    for layer in model.layers[:-5]:
        layer.trainable = False
        
    # Compile the model.
    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    # Data augmentation.
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)
    
    # Train only the top layers first.
    print("Training head only...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=30,
        validation_data=(x_test, y_test)
    )
    
    # Unfreeze all layers and fine-tune.
    for layer in model.layers:
        layer.trainable = True
        
    # Recompile with a lower learning rate.
    optimizer_finetune = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer_finetune, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Fine-tuning the entire network...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(x_test, y_test)
    )
    
    # Evaluate the model.
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Final test loss: {loss:.4f}, Final test accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
