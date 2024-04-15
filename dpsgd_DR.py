import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import time
import numpy as np
import pickle
import argparse
import json

def create_mnist_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

@tf.function
def test_step_dpsgd(test_images, test_labels, models, loss_fn, test_losses, test_accuracies):
    for i, model in enumerate(models):
        predictions = model(test_images, training=False)
        loss = loss_fn(test_labels, predictions)
        test_losses[i].update_state(loss)
        test_accuracies[i].update_state(test_labels, predictions)

@tf.function
def train_step_dpsgd(big_batch_images, big_batch_labels, models, mixing_matrix, optimizers, loss_fn, train_losses, train_accuracies):
    batch_size_per_agent = tf.shape(big_batch_images)[0] // len(models)
    for i, model in enumerate(models):
        start = i * batch_size_per_agent
        end = start + batch_size_per_agent
        images = big_batch_images[start:end]
        labels = big_batch_labels[start:end]
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizers[i].apply_gradients(zip(gradients, model.trainable_variables))
        train_losses[i].update_state(loss)
        train_accuracies[i].update_state(labels, predictions)
    aggregate_parameters(models, mixing_matrix)

@tf.function
def aggregate_parameters(models, mixing_matrix):
    num_agents = len(models)
    for var_idx in range(len(models[0].trainable_variables)):
        vars_to_aggregate = [model.trainable_variables[var_idx] for model in models]
        stacked_vars = tf.stack(vars_to_aggregate, axis=0)
        mixing_matrix_float32 = tf.cast(mixing_matrix, tf.float32)
        weighted_vars = tf.tensordot(mixing_matrix_float32, stacked_vars, axes=[[1], [0]])
        for i, model in enumerate(models):
            model.trainable_variables[var_idx].assign(weighted_vars[i])

def main(mixing_matrix_path, output_file):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape,test_images.shape )
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    num_agents = 10
    big_batch_size = 64 * num_agents
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(big_batch_size).prefetch(tf.data.AUTOTUNE).cache()
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(big_batch_size).prefetch(tf.data.AUTOTUNE).cache()
    
    with open(mixing_matrix_path, 'rb') as file:
        mixing_matrix = pickle.load(file)

    epochs = 50
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    models = [create_mnist_model() for _ in range(num_agents)]
    optimizers = [SGD(learning_rate=0.02) for _ in range(num_agents)]
    train_losses = [tf.keras.metrics.Mean() for _ in range(num_agents)]
    train_accuracies = [tf.keras.metrics.CategoricalAccuracy() for _ in range(num_agents)]
    test_losses = [tf.keras.metrics.Mean() for _ in range(num_agents)]
    test_accuracies = [tf.keras.metrics.CategoricalAccuracy() for _ in range(num_agents)]

    metrics_history = {'train_loss': [[] for _ in range(num_agents)], 'train_accuracy': [[] for _ in range(num_agents)], 'test_accuracy': [[] for _ in range(num_agents)]}

    for epoch in range(epochs):
        for train_loss, train_accuracy, test_loss, test_accuracy in zip(train_losses, train_accuracies, test_losses, test_accuracies):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
        start_time = time.time()
        for big_batch_images, big_batch_labels in train_dataset:
            train_step_dpsgd(big_batch_images, big_batch_labels, models, mixing_matrix, optimizers, loss_fn, train_losses, train_accuracies)
        for test_batch_images, test_batch_labels in test_dataset:
            test_step_dpsgd(test_batch_images, test_batch_labels, models, loss_fn, test_losses, test_accuracies)
        end_time = time.time()
        for i in range(num_agents):
            metrics_history['train_loss'][i].append(train_losses[i].result().numpy())
            metrics_history['train_accuracy'][i].append(train_accuracies[i].result().numpy())
            metrics_history['test_accuracy'][i].append(test_accuracies[i].result().numpy())
            print(f"Epoch:{epoch+1} - Model {i+1} - Loss: {train_losses[i].result().numpy()}, Accuracy: {test_accuracies[i].result().numpy()}, Time: {end_time - start_time:.2f}s")

    with open(output_file, 'wb') as file:
        pickle.dump(metrics_history, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a specified mixing matrix.')
    parser.add_argument('mixing_matrix_path', type=str, help='Path to the mixing matrix file.')
    parser.add_argument('output_file', type=str, help='Output file for saving the results.')
    
    args = parser.parse_args()
    main(args.mixing_matrix_path, args.output_file)
