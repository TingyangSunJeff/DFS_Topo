import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.datasets import mnist
import numpy as np
import time
import pickle
import argparse

@tf.function
def test_step(images, labels, models, loss_fn, test_accuracies):
    for i, model in enumerate(models):
        predictions = model(images, training=False)
        # loss = loss_fn(labels, predictions)
        # test_losses[i].update_state(loss)
        test_accuracies[i].update_state(labels, predictions)

@tf.function
def train_step_dpsgd(images, labels, models, mixing_matrix, optimizers, loss_fn, train_losses, train_accuracies):
    mixing_matrix = tf.cast(mixing_matrix, tf.float32)  # Ensure mixing_matrix is float32
    for i, model in enumerate(models):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizers[i].apply_gradients(zip(gradients, model.trainable_variables))
        train_losses[i].update_state(loss)
        train_accuracies[i].update_state(labels, predictions)
    # Simulate D-PSGD parameter aggregation after each training step
    aggregate_parameters(models, mixing_matrix)

@tf.function
def aggregate_parameters(models, mixing_matrix):
    num_agents = len(models)
    for var_idx in range(len(models[0].trainable_variables)):
        # Extract the variable from all models to form a list
        vars_to_aggregate = [model.trainable_variables[var_idx] for model in models]
        # Stack the variables along a new dimension to make them a single tensor
        stacked_vars = tf.stack(vars_to_aggregate, axis=0)
        
        # Apply the mixing matrix (weights) to the stacked variables
        # mixing_matrix shape: [num_agents, num_agents]
        # stacked_vars shape: [num_agents, var_shape...]
        # The mixing_matrix is broadcasted to the shape of the variables for weighted sum
        weighted_vars = tf.tensordot(mixing_matrix, stacked_vars, axes=[[1], [0]])
        
        # Assign the weighted sum back to each model's variable
        for i, model in enumerate(models):
            model.trainable_variables[var_idx].assign(weighted_vars[i])
            
# Define the MNIST model
def create_mnist_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a 4-layer CNN model for the MNIST dataset.
    """
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

def load_and_preprocess_data(dataset_name='mnist'):
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # CIFAR-10: Resize and preprocess
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, -1)  # Add channel dimension
        x_test = np.expand_dims(x_test, -1)
        # MNIST: Only normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    else:
        raise ValueError("Unsupported dataset. Please choose either 'cifar10' or 'mnist'.")

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def main(mixing_matrix_path, output_file):
    with open(mixing_matrix_path, 'rb') as file:
        # Load the content of the file into a Python object
        mixing_matrix = pickle.load(file)
    # Replace 'yourfile.pkl' with the path to your pickle file
    num_agents = 10
    # mixing_matrix = np.full((num_agents, num_age
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    models = [create_mnist_model((28, 28, 1), 10) for _ in range(num_agents)]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(64).prefetch(tf.data.experimental.AUTOTUNE)
    optimizers = [tf.keras.optimizers.Adam(learning_rate=0.0001) for _ in range(num_agents)]
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    train_losses, train_accuracies, test_accuracies = [], [], []
    
    for i in range(num_agents):
        train_losses.append(tf.keras.metrics.Mean(name=f'train_loss_{i}'))
        train_accuracies.append(CategoricalAccuracy(name=f'train_accuracy_{i}'))
        test_accuracies.append(CategoricalAccuracy(name=f'test_accuracy_{i}'))

    epochs = 50
    metrics_history = {
        'train_loss': [[] for _ in range(num_agents)],
        'test_accuracy': [[] for _ in range(num_agents)],
    }

    for epoch in range(epochs):
        start_time = time.time()
        # Reset metrics at the start of each epoch
        for train_loss, train_accuracy, test_accuracy in zip(train_losses, train_accuracies, test_accuracies):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
        # Training step
        for images, labels in train_ds:
            train_step_dpsgd(images, labels, models, mixing_matrix, optimizers, loss_fn, train_losses, train_accuracies)
        # Testing step
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, models, loss_fn, test_accuracies)
        end_time = time.time()

        # Save metrics for this epoch
        for i in range(num_agents):
            metrics_history['train_loss'][i].append(train_losses[i].result().numpy())
            metrics_history['test_accuracy'][i].append(test_accuracies[i].result().numpy())
            # Optionally print the metrics
            print(f"Agent {i+1}, Epoch {epoch+1}, Loss: {train_losses[i].result()}, Accuracy: {train_accuracies[i].result()*100}%, Time: {end_time - start_time:.2f}s")

    with open(output_file, 'wb') as file:
        pickle.dump(metrics_history, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a specified mixing matrix.')
    parser.add_argument('mixing_matrix_path', type=str, help='Path to the mixing matrix file.')
    parser.add_argument('output_file', type=str, help='Output file for saving the results.')
    
    args = parser.parse_args()
    main(args.mixing_matrix_path, args.output_file)
