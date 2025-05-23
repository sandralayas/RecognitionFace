# # Face recognition dataset Labelled Faces in the Wild

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("stoicstatic/face-recognition-dataset")

# print("Path to dataset files:", path)

# # This is the response / answer of the following stack-overflow question: https://stackoverflow.com/q/79404980/9215780

import os
os.environ["KERAS_BACKEND"] = "tensorflow" # tensorflow, torch, jax

import zipfile
import random
import math
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from keras import Model
from keras.layers import (
    Layer, Flatten, Dense, Dropout, BatchNormalization, Input
)
from keras.metrics import Mean, CosineSimilarity
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.applications import EfficientNetB7

import torch
import tensorflow as tf

keras.config.backend()

FACE_DATA_PATH = "C:/Users/sandr/.cache/kagglehub/datasets/stoicstatic/face-recognition-dataset/versions/10/Face Data/Face Dataset"
EXTRACTED_FACES_PATH = "C:/Users/sandr/.cache/kagglehub/datasets/stoicstatic/face-recognition-dataset/versions/10/Extracted Faces/Extracted Faces"

import shutil

DATASET = 'images/output_dataset'
if os.path.exists(DATASET):
    shutil.rmtree(DATASET)    
os.makedirs(DATASET)

def copy_to_output_dataset(input_path, output_path):
    for person_folder in os.listdir(input_path):
        person_folder_path = os.path.join(input_path, person_folder)
        if os.path.isdir(person_folder_path):
            output_person_folder = os.path.join(output_path, person_folder)
            if not os.path.exists(output_person_folder):
                os.makedirs(output_person_folder)

            for image_file in os.listdir(person_folder_path):
                if image_file.endswith('.jpg'):
                    src_image_path = os.path.join(person_folder_path, image_file)
                    dst_image_path = os.path.join(output_person_folder, image_file)
                    if os.path.exists(dst_image_path):
                        base, ext = os.path.splitext(dst_image_path)
                        dst_image_path = f"{base}_1{ext}"
                    shutil.copy(src_image_path, dst_image_path)

copy_to_output_dataset(FACE_DATA_PATH, DATASET)                
copy_to_output_dataset(EXTRACTED_FACES_PATH, DATASET) 

def triplets(folder_paths, max_triplets=7):
    anchor_images = []
    positive_images = []
    negative_images = []

    for person_folder in folder_paths:
        images = [os.path.join(person_folder, img)
                  for img in os.listdir(person_folder)]
        num_images = len(images)

        if num_images < 2:
            continue

        random.shuffle(images)

        for _ in range(max(num_images-1, max_triplets)):
            anchor_image = random.choice(images)

            positive_image = random.choice(
                [x for x in images if x != anchor_image]
            )

            negative_folder = random.choice(
                [x for x in folder_paths if x != person_folder]
            )

            negative_image = random.choice(
                [os.path.join(negative_folder, img) for img in os.listdir(negative_folder)]
            )

            anchor_images.append(anchor_image)
            positive_images.append(positive_image)
            negative_images.append(negative_image)

    return anchor_images, positive_images, negative_images

person_folders = [
    os.path.join(DATASET, folder_name) 
    for folder_name in os.listdir(DATASET)
]

anchors, positives, negatives = triplets(person_folders)

def split_triplets(
    anchors,
    positives,
    negatives,
    validation_split=0.2
):
    triplets = list(zip(anchors, positives, negatives))
    train_triplets, val_triplets = train_test_split(
        triplets,
        test_size=validation_split,
        random_state=42
    )
    return train_triplets, val_triplets

train_triplets, val_triplets = split_triplets(
    anchors,
    positives,
    negatives
)
len(train_triplets), len(val_triplets)

def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    return image

def load_and_preprocess_triplet(triplet):
    anchor = triplet[0]
    positive = triplet[1]
    negative = triplet[2]
    return (
        load_and_preprocess_image(anchor),
        load_and_preprocess_image(positive),
        load_and_preprocess_image(negative)
    )

augmentation_layers = [
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomZoom(.5, .2),
    keras.layers.RandomRotation(0.3)
]

def augment_triplet(anchor, positive, negative):

    for layer in augmentation_layers:
        anchor = layer(anchor)
        positive = layer(positive)
        negative = layer(negative)
    
    return anchor, positive, negative

def data_generator(samples, batch_size, shuffle=True, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(samples)
    dataset = dataset.map(
        load_and_preprocess_triplet, num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    
    if augment:
        dataset = dataset.map(
            augment_triplet, num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


batch_size = 16
train_dataset = data_generator(
    samples=train_triplets,
    batch_size=batch_size,
    shuffle=True,
    augment=True
)
val_dataset = data_generator(
    samples=val_triplets,
    batch_size=batch_size,
    shuffle=False,
    augment=False
)

def visualize_triplets(triplets, num_examples):
    anchor_batch, positive_batch, negative_batch = triplets

    for i in range(len(anchor_batch)):

        if i > num_examples-1:
            break
        
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Anchor")
        plt.imshow(anchor_batch[i] / 255.)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Positive")
        plt.imshow(positive_batch[i] / 255.)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Negative")
        plt.imshow(negative_batch[i] / 255.)
        plt.axis('off')

        plt.show()

batch = next(iter(train_dataset))

visualize_triplets(batch, num_examples=5)

def get_embedding(input_shape, num_layers_to_unfreeze=25):
    base_model = EfficientNetB7(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False,
        pooling='avg'
    )

    for i in range(len(base_model.layers)-num_layers_to_unfreeze):
        base_model.layers[i].trainable = False

    embedding = keras.Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128)
    ], name='Embedding')

    return embedding

input_shape = (128, 128, 3)
embedding = get_embedding(input_shape)
embedding.summary()

from keras import ops

class DistanceLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = ops.sum(ops.square(anchor - positive), -1)
        an_distance = ops.sum(ops.square(anchor - negative), -1)
        return ap_distance, an_distance

anchor_input = Input(name='anchor', shape=input_shape)
positive_input = Input(name='positive', shape=input_shape)
negative_input = Input(name='negative', shape=input_shape)

distances = DistanceLayer()(
    embedding(anchor_input),
    embedding(positive_input),
    embedding(negative_input)
)

siamese_net = Model(
    inputs=[anchor_input,
            positive_input,
            negative_input],
    outputs=distances
)

siamese_net.summary()

plot_model(siamese_net, show_shapes=True, show_layer_names=True)

class SiameseModel(Model):
    def __init__(self, siamese_net, margin=0.5):
        super().__init__()
        self.siamese_net = siamese_net
        self.margin = margin
        self.loss_tracker = Mean(name='loss')
        self.accuracy_tracker = Mean(name='accuracy')

    def call(self, inputs):
        return self.siamese_net(inputs)

    def train_step(self, *args, **kwargs):
        if keras.backend.backend() == "jax":
            return self._jax_train_step(*args, **kwargs)
        elif keras.backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.backend.backend() == "torch":
            return self._torch_train_step(*args, **kwargs)

    def _jax_train_step(self, state, data):
        raise NotImplementedError("JAX backend is not supported for this model.")
    
    def _torch_train_step(self, data):
        self.zero_grad()
        ap_distance, an_distance = self(data)
        loss = self._compute_loss(ap_distance, an_distance)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self.loss_tracker.update_state(loss)
        accuracy = self._compute_accuracy(ap_distance, an_distance)
        self.accuracy_tracker.update_state(accuracy)
        return {m.name: m.result() for m in self.metrics}

    def _tensorflow_train_step(self, data):
        with tf.GradientTape() as tape:
            ap_distance, an_distance = self(data)
            loss = self._compute_loss(ap_distance, an_distance)

        gradients = tape.gradient(loss, self.siamese_net.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_net.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        accuracy = self._compute_accuracy(ap_distance, an_distance)
        self.accuracy_tracker.update_state(accuracy)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        ap_distance, an_distance = self(data)
        loss = self._compute_loss(ap_distance, an_distance)
        accuracy = self._compute_accuracy(ap_distance, an_distance)
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(accuracy)
        return {m.name: m.result() for m in self.metrics}

    def _compute_loss(self, ap_distance, an_distance):
        loss = ap_distance - an_distance
        loss = ops.maximum(loss + self.margin, .0)
        return ops.mean(loss)

    def _compute_accuracy(self, ap_distance, an_distance):
        accuracy = ops.mean(
            ops.cast(ap_distance < an_distance, 'float32')
        )
        return accuracy

    @property
    def metrics(self):
        return super().metrics + self.train_metrics
    
    @property
    def train_metrics(self):
        return [
            self.loss_tracker, 
            self.accuracy_tracker, 
        ]

# Create a tensor
x = torch.randn(2, 3)

# Check its device
print(x.device)

siamese_model = SiameseModel(siamese_net)
_ = siamese_model(batch)
siamese_model.compile(optimizer=Adam(0.001))

siamese_model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    callbacks=[keras.callbacks.CSVLogger('history.csv')],
    epochs=5
) 

history = pd.read_csv('history.csv')

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['TRAIN', 'VAL'], loc='lower right')

plt.subplot(122)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['TRAIN', 'VAL'], loc='lower right')

plt.show()

sample = next(iter(val_dataset))
anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(anchor),
    embedding(positive),
    embedding(negative)
)
cosine_similarity = CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print(f'Positive similarity: {positive_similarity}')

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print(f'Negative similarity: {negative_similarity}')

