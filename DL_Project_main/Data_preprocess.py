import os
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class preprocess_data:
    def visualization_images(self, dir_path, nimages):
        fig, axs = plt.subplots(2, 4, figsize=(8, 4))
        dpath = dir_path
        count = 0
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in range(nimages):
                img_path = os.path.join(dpath, i, train_class[j])
                img = cv2.imread(img_path)
                axs[count][j].set_title(i)
                axs[count][j].imshow(img)
            count += 1
            print(img)
        fig.tight_layout()
        plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train imaegs :{}\n'.format(len(train)))
        print('Numeber of train images labels:{}\n'.format(len(label)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})

        return train, label, retina_df

    def generate_train_test_images(self, train, label):
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        train_data, test_data = train_test_split(retina_df, test_size=0.2)
        print(test_data)
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            validation_split=0.15
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='training'
        )
        validation_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32,
            subset='validation'
        )
        test_generator = test_datagen.flow_from_dataframe(
            test_data,
            directory='./',
            x_col="Image",
            y_col="Labels",
            target_size=(28, 28),
            color_mode="rgb",
            class_mode="categorical",
            batch_size=32
        )
        sample_images, sample_labels = train_generator.next()
        img_shape = sample_images[0].shape
        print('--------------------------------')
        print(f'image shape: {img_shape}')
        print(f"Train images shape:{train_data.shape}")
        print(f"Testing images shape:{test_data.shape}")
        return train_generator, test_generator, validation_generator

    def plot_history(self,history):
        #loss
        plt.plot(history.history['loss'],label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=True)
        #accuracy
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show(block=True)