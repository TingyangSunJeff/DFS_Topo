#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import argparse
import time

def resnet_block(x, filters, strides=1, projection_shortcut=False):
    """A basic residual block for CIFAR-10 ResNet.
    
    Args:
      x: Input tensor.
      filters: Number of filters for the convolutions.
      strides: Stride for the first convolution.
      projection_shortcut: If True, use a convolution shortcut to match dimensions.
    
    Returns:
      Output tensor after applying the block.
    """
    shortcut = x
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    # If dimensions differ, use a projection shortcut.
    if projection_shortcut:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add the shortcut connection and apply ReLU.
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def create_resnet20_model(input_shape=(32, 32, 3), num_classes=10):
    """Creates a ResNet20 model for CIFAR-10.
    
    The architecture follows:
      - An initial 3x3 conv with 16 filters.
      - 3 stages of residual blocks:
          * Stage 1: 3 blocks with 16 filters.
          * Stage 2: 3 blocks with 32 filters (first block downsamples).
          * Stage 3: 3 blocks with 64 filters (first block downsamples).
      - Global average pooling and a final dense layer.
    
    Args:
      input_shape: Input shape of images.
      num_classes: Number of output classes.
    
    Returns:
      A Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape)
    # Initial conv layer.
    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Stage 1: 3 residual blocks with 16 filters.
    for _ in range(3):
        x = resnet_block(x, filters=16, strides=1, projection_shortcut=False)
    
    # Stage 2: 3 blocks with 32 filters; first block downsamples.
    x = resnet_block(x, filters=32, strides=2, projection_shortcut=True)
    for _ in range(2):
        x = resnet_block(x, filters=32, strides=1, projection_shortcut=False)
    
    # Stage 3: 3 blocks with 64 filters; first block downsamples.
    x = resnet_block(x, filters=64, strides=2, projection_shortcut=True)
    for _ in range(2):
        x = resnet_block(x, filters=64, strides=1, projection_shortcut=False)
    
    # Global average pooling and classification layer.
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

@tf.function
def train_step_dpsgd(big_batch_images, big_batch_labels, models_list, mixing_matrix,
                     optimizers, loss_fn, train_loss_metrics, train_accuracy_metrics):
    """
    Performs one DPSGD training step.
    Splits a large batch among agents, applies local gradient updates,
    then aggregates parameters using the mixing matrix.
    """
    num_agents = len(models_list)
    batch_size = tf.shape(big_batch_images)[0]
    per_agent = batch_size // num_agents
    for i in range(num_agents):
        start = i * per_agent
        end = start + per_agent
        images = big_batch_images[start:end]
        labels = big_batch_labels[start:end]
        with tf.GradientTape() as tape:
            predictions = models_list[i](images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, models_list[i].trainable_variables)
        optimizers[i].apply_gradients(zip(gradients, models_list[i].trainable_variables))
        train_loss_metrics[i].update_state(loss)
        train_accuracy_metrics[i].update_state(labels, predictions)
    
    # Aggregate parameters across agents.
    aggregate_parameters(models_list, mixing_matrix)

@tf.function
def test_step(models_list, images, labels, loss_fn, test_loss_metrics, test_accuracy_metrics):
    """
    Evaluates the models on a test batch.
    Each agent evaluates the same batch.
    """
    num_agents = len(models_list)
    for i in range(num_agents):
        predictions = models_list[i](images, training=False)
        loss = loss_fn(labels, predictions)
        test_loss_metrics[i].update_state(loss)
        test_accuracy_metrics[i].update_state(labels, predictions)

@tf.function
def aggregate_parameters(models_list, mixing_matrix):
    """
    Aggregates each trainable variable from all agent models
    using the provided mixing matrix.
    """
    num_agents = len(models_list)
    for var_idx in range(len(models_list[0].trainable_variables)):
        var_list = [model.trainable_variables[var_idx] for model in models_list]
        stacked_vars = tf.stack(var_list, axis=0)
        mixing_matrix_float = tf.cast(mixing_matrix, tf.float32)
        aggregated_vars = tf.tensordot(mixing_matrix_float, stacked_vars, axes=[[1], [0]])
        for i, model in enumerate(models_list):
            model.trainable_variables[var_idx].assign(aggregated_vars[i])

def main(mixing_matrix_path, output_file):
    # Load CIFAR-10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Hyperparameters.
    num_agents = 10
    big_batch_size = 64 * num_agents
    epochs = 100

    # Learning rate schedule.
    steps_per_epoch = x_train.shape[0] // big_batch_size
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[30 * steps_per_epoch, 60 * steps_per_epoch],
        values=[0.8, 0.08, 0.01]
    )

    # Load the mixing matrix from the provided pickle file.
    with open(mixing_matrix_path, "rb") as f:
        mixing_matrix = pickle.load(f)
    # Create tf.data datasets.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(big_batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(big_batch_size).prefetch(tf.data.AUTOTUNE)

    # Create models and optimizers for each agent using ResNet20.
    models_list = [create_resnet20_model() for _ in range(num_agents)]
    optimizers = [tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
                  for _ in range(num_agents)]

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Create metrics for each agent.
    train_loss_metrics = [tf.keras.metrics.Mean() for _ in range(num_agents)]
    train_accuracy_metrics = [tf.keras.metrics.CategoricalAccuracy() for _ in range(num_agents)]
    test_loss_metrics = [tf.keras.metrics.Mean() for _ in range(num_agents)]
    test_accuracy_metrics = [tf.keras.metrics.CategoricalAccuracy() for _ in range(num_agents)]

    metrics_history = {
        "train_loss": [[] for _ in range(num_agents)],
        "train_accuracy": [[] for _ in range(num_agents)],
        "test_loss": [[] for _ in range(num_agents)],
        "test_accuracy": [[] for _ in range(num_agents)]
    }

    # Main training loop.
    for epoch in range(epochs):
        start_time = time.time()
        for metric in train_loss_metrics + train_accuracy_metrics + test_loss_metrics + test_accuracy_metrics:
            metric.reset_states()

        for big_batch_images, big_batch_labels in train_dataset:
            train_step_dpsgd(big_batch_images, big_batch_labels, models_list, mixing_matrix,
                             optimizers, loss_fn, train_loss_metrics, train_accuracy_metrics)
        
        for test_images, test_labels in test_dataset:
            test_step(models_list, test_images, test_labels, loss_fn, test_loss_metrics, test_accuracy_metrics)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} - Time: {epoch_time:.2f}s")
        for i in range(num_agents):
            train_loss_val = train_loss_metrics[i].result().numpy()
            train_acc_val = train_accuracy_metrics[i].result().numpy()
            test_loss_val = test_loss_metrics[i].result().numpy()
            test_acc_val = test_accuracy_metrics[i].result().numpy()
            metrics_history["train_loss"][i].append(train_loss_val)
            metrics_history["train_accuracy"][i].append(train_acc_val)
            metrics_history["test_loss"][i].append(test_loss_val)
            metrics_history["test_accuracy"][i].append(test_acc_val)
            print(f"  Agent {i+1}: Train Loss: {train_loss_val:.4f}, Train Acc: {train_acc_val:.4f} | "
                  f"Test Loss: {test_loss_val:.4f}, Test Acc: {test_acc_val:.4f}")

    # Save training history.
    with open(output_file, "wb") as f:
        pickle.dump(metrics_history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPSGD Training with ResNet20 on CIFAR-10")
    parser.add_argument("mixing_matrix_path", type=str, help="Path to the mixing matrix file (pickle format)")
    parser.add_argument("output_file", type=str, help="Output file to save training metrics (pickle format)")
    args = parser.parse_args()
    main(args.mixing_matrix_path, args.output_file)
