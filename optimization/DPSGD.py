# training/p_psgd_training.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
import os
from config import PROJECT_PATH
from tensorflow.keras.datasets import mnist, cifar10
from config import OVERLAY_NODES

def load_and_preprocess_data(dataset_name='mnist'):
    """
    Load and preprocess dataset.

    Parameters:
    - dataset_name: A string, either 'mnist' or 'cifar10' specifying which dataset to load.

    Returns:
    - Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    if dataset_name == 'mnist':
        dataset_path = os.path.join(PROJECT_PATH, 'mnist.npz')
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path=dataset_path)
        x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
        x_train = x_train[..., np.newaxis]  # Add channel dimension
        x_test = x_test[..., np.newaxis]
    elif dataset_name == 'cifar10':
        dataset_path = os.path.join(PROJECT_PATH, 'cifar10.npz')
        (x_train, y_train), (x_test, y_test) = cifar10.load_data(path=dataset_path)
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


def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10)
    ])
    return model

def d_psgd_training(x_train, y_train, x_test, y_test, mixing_matrix, epochs=80, batch_size=64):
    # Initialize models and optimizers for each agent
    num_agents = len(OVERLAY_NODES)
    agents_models = [create_model() for _ in range(num_agents)]
    optimizers = [tf.keras.optimizers.SGD(learning_rate=0.01) for _ in range(num_agents)]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
                    print(f"Step {step}: Loss = {loss.numpy()}")
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

def evaluate_agents_models(agents_models, x_test, y_test, batch_size=64):
    # Evaluation logic as in your script
    pass
