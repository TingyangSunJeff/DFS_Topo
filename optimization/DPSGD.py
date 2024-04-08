# training/p_psgd_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import mnist, cifar10

def load_and_preprocess_data(dataset_name='mnist'):
    """
    Load and preprocess dataset.

    Parameters:
    - dataset_name: A string, either 'mnist' or 'cifar10' specifying which dataset to load.

    Returns:
    - Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    # current_dir = os.getcwd()
    if dataset_name == 'mnist':
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
        x_train = x_train[..., np.newaxis]  # Add channel dimension
        x_test = x_test[..., np.newaxis]
    elif dataset_name == 'cifar10':
        # Load the CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
    else:
        raise ValueError("Unsupported dataset. Please choose either 'mnist' or 'cifar10'.")


    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)  # Ensure labels are integers
    
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Training set shape: {x_train.shape}")
    print(f"Test set shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Training set dtype: {x_train.dtype}")
    print(f"Test set dtype: {x_test.dtype}")
    print(f"Min and max training values: {x_train.min()}, {x_train.max()}")
    
    return (x_train, y_train), (x_test, y_test)


def create_mnist_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a 4-layer CNN model for the MNIST dataset.
    
    Parameters:
    - input_shape: The shape of the input data, including the channel dimension.
    - num_classes: The number of classes for the output layer.
    
    Returns:
    A TensorFlow Keras model instance.
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

def create_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10)
    ])
    return model

def create_resnet50_model(input_shape, num_classes):
    """
    Create a ResNet50 model for classification.

    Parameters:
    - input_shape: The shape of the input data (height, width, channels).
    - num_classes: The number of classes for the output layer.

    Returns:
    A TensorFlow Keras model instance.
    """
    # Load the ResNet50 model, excluding its top (output) layer
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Set all layers of the base_model to be trainable
    base_model.trainable = False

    # Add new layers on top of the model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def d_psgd_training(x_train, y_train, x_test, y_test, mixing_matrix, num_agents, dataset_flag, epochs=80, batch_size=64):
    """
        Train models using decentralized parallel SGD with a specified dataset.
        
        Parameters:
        - x_train, y_train: Training data and labels.
        - x_test, y_test: Test data and labels.
        - mixing_matrix: The mixing matrix W used for parameter aggregation.
        - num_agents: The number of agents (models) to train in parallel.
        - dataset_flag: A flag indicating which dataset (and model) to use ('mnist' or 'cifar10').
        - epochs: The number of training epochs.
        - batch_size: The size of the batches for training.
    """
    # Initialize models based on the dataset
    if dataset_flag == 'mnist':
        agents_models = [create_model(input_shape=(28, 28, 1), num_classes=10) for _ in range(num_agents)]
    elif dataset_flag == 'cifar10':
        agents_models = [create_resnet50_model(input_shape=(32, 32, 3), num_classes=10) for _ in range(num_agents)]
    else:
        raise ValueError("Unsupported dataset flag. Please choose either 'mnist' or 'cifar10'.")

    optimizers = [tf.keras.optimizers.SGD(learning_rate=0.02) for _ in range(num_agents)]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Define the mixing matrix W and other training parameters here
    # Followed by the training loop and evaluation as in your script
    # Example of tracking accuracy
    lost_hitory = []
    val_accuracies = []
    # Custom training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # Reset epoch loss for each agent
        epoch_loss = [0] * num_agents
        # Shuffle and batch the data
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(2000).batch(batch_size)
        for step, (x_batch, y_batch) in enumerate(dataset):
            for i in range(num_agents):
                # Aggregation of parameters based on W
                aggregated_params = []
                for var_idx, _ in enumerate(agents_models[i].trainable_variables):
                    weighted_sum = sum(mixing_matrix[i, j] * agents_models[j].trainable_variables[var_idx] for j in range(num_agents))
                    aggregated_params.append(weighted_sum)

                # Applying the gradient descent step
                with tf.GradientTape() as tape:
                    # Compute loss using the original parameters
                    predictions = agents_models[i](x_batch, training=True)
                    loss = loss_fn(y_batch, predictions)
                # Update epoch loss for agent i
                epoch_loss[i] += loss.numpy()
                # Compute gradients with respect to the original parameters
                grads = tape.gradient(loss, agents_models[i].trainable_variables)

                # update model parameters to the aggregated ones for gradient update
                for var, new_val in zip(agents_models[i].trainable_variables, aggregated_params):
                    var.assign(new_val)
                # Update parameters using the computed gradients
                optimizers[i].apply_gradients(zip(grads, agents_models[i].trainable_variables))

                if step % 100 == 0:
                    print(f"Step {step} {i+1}/{num_agents}: Loss = {loss.numpy()}")

                    
        lost_hitory.append(sum(epoch_loss)/len(epoch_loss))
        print(f"Epoch {epoch+1}, global objective loss: {sum(epoch_loss)/len(epoch_loss)}")
        # Validation step at the end of each epoch
        total_correct = 0
        total_val_samples = 0
        for x_batch_val, y_batch_val in test_dataset:
            val_predictions = [agents_models[i](x_batch_val, training=False) for i in range(num_agents)]
            # Assuming a strategy to aggregate predictions from all agents, e.g., averaging
            avg_predictions = tf.reduce_mean(val_predictions, axis=0)
            correct_preds = tf.equal(tf.argmax(avg_predictions, axis=1, output_type=tf.int32), y_batch_val)
            total_correct += tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_val_samples += x_batch_val.shape[0]
        val_accuracy = total_correct / total_val_samples
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy.numpy()}")
    return lost_hitory, val_accuracies