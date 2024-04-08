import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from multiprocessing import Process
import os

# Data preprocessing
def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)
    label = tf.one_hot(label, depth=10)
    return image, label

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset



def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model_on_gpu(gpu_id, model, train_dataset, epochs=5):
    with tf.device(f'/gpu:{gpu_id}'):
        for epoch in range(epochs):
            model.fit(train_dataset, epochs=1, verbose=1)
            print(f'Model on GPU {gpu_id} - Epoch {epoch+1} completed')


def train_on_gpu(gpu_id):
    print(f"Training on GPU {gpu_id} with PID {os.getpid()}")

    # Set the GPU to use
    tf.config.set_visible_devices([], 'GPU')  # Disable all GPUs initially
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            assert len(logical_devices) == 1
            print(f"Using GPU {gpu_id}")
        except RuntimeError as e:
            print(e)

    # Load and preprocess data (this could be moved outside and data passed as an argument)
    train_dataset, test_dataset = load_and_preprocess_data()

    # Model creation and training
    model = create_model()
    model.fit(train_dataset, epochs=5, verbose=1, validation_data=test_dataset)
    print(f"Training completed on GPU {gpu_id}")

if __name__ == '__main__':
    processes = []

    # Create a process for each model/GPU pair
    for gpu_id in range(3):
        p = Process(target=train_on_gpu, args=(gpu_id,))
        p.start()
        processes.append(p)