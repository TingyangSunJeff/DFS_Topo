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
    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor, pooling='avg')
    x = Flatten()(base_model.output)
    output_tensor = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

@tf.function
def train_step_dpsgd(big_batch_images, big_batch_labels, models, mixing_matrix, optimizers, loss_fn, train_losses, train_accuracies):
    # Split the big batch into smaller batches for each agent
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
        mixing_matrix_float32 = tf.cast(mixing_matrix, tf.float32)
        weighted_vars = tf.tensordot(mixing_matrix_float32, stacked_vars, axes=[[1], [0]])
        
        # Assign the weighted sum back to each model's variable
        for i, model in enumerate(models):
            model.trainable_variables[var_idx].assign(weighted_vars[i])


def main(mixing_matrix_path, output_file):
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    print("=========", mixing_matrix_path)
    # Normalize pixel values
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # batch_size = 64
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE).cache()
    num_agents = 10
    big_batch_size = 64 * num_agents
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(big_batch_size).prefetch(tf.data.AUTOTUNE).cache()
    # Open the file in binary read mode
    with open(mixing_matrix_path, 'rb') as file:
        # Load the content of the file into a Python object
        mixing_matrix = pickle.load(file)

    # model = create_resnet50_cifar10()
    # total_params = model.count_params()
    # print(f'Total Parameters: {total_params}')
    epochs = 30
    num_agents = 10
    # Loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    models = [create_resnet50_cifar10() for _ in range(num_agents)]
    optimizers = [tf.keras.optimizers.Adam() for _ in range(num_agents)] # 0.001 learning rate
    train_losses = [tf.keras.metrics.Mean() for _ in range(num_agents)]
    train_accuracies = [tf.keras.metrics.CategoricalAccuracy() for _ in range(num_agents)]
    metrics_history = {
        'train_loss': [[] for _ in range(num_agents)],
        'test_accuracy': [[] for _ in range(num_agents)]
    }
    for epoch in range(epochs):
        # Training loop for one epoch
        for train_loss, train_accuracy in zip(train_losses, train_accuracies):
            train_loss.reset_states()
            train_accuracy.reset_states()
        start_time = time.time()
        for big_batch_images, big_batch_labels in train_dataset:
            train_step_dpsgd(big_batch_images, big_batch_labels, models, mixing_matrix, optimizers, loss_fn, train_losses, train_accuracies)
        end_time = time.time()
        # Print training loss and accuracy
        for i in range(num_agents):
            metrics_history['train_loss'][i].append(train_losses[i].result().numpy())
            metrics_history['test_accuracy'][i].append(train_accuracies[i].result().numpy())
            print(f"Epoch:{epoch+1} - Model {i+1} - Loss: {train_losses[i].result().numpy()}, Accuracy: {train_accuracies[i].result().numpy()}, Time: {end_time - start_time:.2f}s")

    # After training is complete, save the model weights
    # for i, model in enumerate(models):
    #     model_weights_path = f"./model_weights/model_{i+1}_weights_epoch_{epochs}.tf"
    #     model.save_weights(model_weights_path)
    #     metrics_history['model_weights_paths'][i] = model_weights_path

    with open(output_file, 'wb') as file:
        pickle.dump(metrics_history, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with a specified mixing matrix.')
    parser.add_argument('mixing_matrix_path', type=str, help='Path to the mixing matrix file.')
    parser.add_argument('output_file', type=str, help='Output file for saving the results.')
    
    args = parser.parse_args()
    main(args.mixing_matrix_path, args.output_file)