import os
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.src.utils import to_categorical

import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf



import Data_preprocess as dp
import Plant_models as cm


def load_images_in_batches(image_paths, batch_size):
    num_batches = len(image_paths) // batch_size
    for i in range(num_batches):
        batch_paths = image_paths[i*batch_size : (i+1)*batch_size]
        batch_images = np.array([cv2.imread(img) for img in batch_paths])
        yield batch_images

def CNN_Call(cnn_model):
    cnn_model.fit(train_generator, epochs=5, validation_data=validation_generator)
    Cnn_test_loss, Cnn_test_acc = cnn_model.evaluate(test_generator)
    print(f'CNN Test accuracy:{Cnn_test_acc}')
    print(cnn_model.summary())

def visualization_images(images_folder_path, param):
    imdata = dp.preprocess_data()
    imdata.visualization_images(images_folder_path, param)


def preprocess_images(images):
     processed_images = []
     for img in images:
         img = cv2.resize(img, (28, 28))
         img = img / 255.0
         processed_images.append(img)
     return np.array(processed_images)
def random_mini_batch(X, Y, mini_batch_size=64, seed=None):
     if seed is not None:
        np.random.seed(seed)
     m = X.shape[0] if len(X.shape) > 1 else len(X)
     mini_batches = []
     permutation = list(np.random.permutation(m))
     shuffled_X = X[permutation, ...]
     shuffled_Y = Y[permutation, ...]
     num_complete_minibatches = m // mini_batch_size
     for k in range(0, num_complete_minibatches):
         start_idx = k * mini_batch_size
         end_idx = (k + 1) * mini_batch_size
         mini_batch_X = shuffled_X[start_idx:end_idx, ...]
         mini_batch_Y = shuffled_Y[start_idx:end_idx, ...]
         mini_batch = (mini_batch_X, mini_batch_Y)
         mini_batches.append(mini_batch)
     if m % mini_batch_size != 0:
         start_idx=num_complete_minibatches*mini_batch_size
         end_idx=start_idx+mini_batch_size
         mini_batch_X = shuffled_X[start_idx:end_idx, ...]
         mini_batch_Y = shuffled_Y[start_idx:end_idx, ...]
         mini_batch = (mini_batch_X, mini_batch_Y)
         mini_batches.append(mini_batch)
     return mini_batches
def one_hot_encode(labels):
     unique_labels=list(set(labels))
     label_dict={label: i for i, label in enumerate(unique_labels)}
     numeric_labels=[label_dict[label] for label in labels]
     one_hot_labels = to_categorical(numeric_labels) # Pass num_classes to to_categorical
     return one_hot_labels


if __name__ == "__main__":
    images_folder_path = 'train'
    imdata = dp.preprocess_data()
    # Assuming you want to visualize 4 images here
    imdata.visualization_images(images_folder_path, 4)
    train, label,image_df = imdata.preprocess(images_folder_path)
    # image_df.to_csv("skill.csv")
    train_generator, test_generator, validation_generator=imdata.generate_train_test_images(train,label)

    def option1():
        print("ANN Model")
        AnnModel = cm.DeepANN()
        Model1 = AnnModel.simple_model()
        print("train generator", train_generator)
        ANN_history = Model1.fit(train_generator, epochs=50, validation_data=validation_generator)

        Ann_test_loss, Ann_test_acc = Model1.evaluate(test_generator)
        print(f'Test Accuracy: {Ann_test_acc}')
        print(f'Test Loss: {Ann_test_loss}')

        Model1.save("my_model1.h5")
        print("the ann architecture is")
        print(Model1.summary())
        print("plot the graph")
        imdata.plot_history(ANN_history)

        # image_shape = (28, 28, 3)
        # model_adam = cm.DeepANN().simple_model(image_shape, optimizer='adam')
        # model_sgd = cm.DeepANN().simple_model(image_shape, optimizer='sgd')
        # model_rmsprop = cm.DeepANN().simple_model(image_shape, optimizer='rmsprop')
        #
        # cm.compare_models([model_adam, model_sgd, model_rmsprop], train_generator, validation_generator, epochs=3)
        # my_model_test_loss, my_model_test_acc = model_adam.evaluate(test_generator)
        # print(f'Test accuracy: {my_model_test_acc}')


    def option2():
        print("CNN Model")
        # CNN Model
        Cnn_model = cm.CNN.simple_cnn()
        CNN_Call(Cnn_model)
        Cnn_batch = cm.CNN.cnn_batch()
        CNN_Call(Cnn_batch)


    def option3():
        # print("Mini-batch")
        # X_train_images = np.array([cv2.imread(img) for img in train])
        # X_train_images = preprocess_images(X_train_images)
        # Y_train = label
        # Y_train_numeric = one_hot_encode(Y_train)
        # batch_size=64
        # image_shape=(28,28,3)
        # for batch_images in load_images_in_batches(train, batch_size):
        #     X_train_images = batch_images
        #     Y_train = label[:len(batch_images)]  # Assuming labels correspond to batch images
        #     Y_train_numeric = one_hot_encode(Y_train)
        #     # Further processing with mini-batches
        #     print("Mini-batch X shape:", batch_images.shape)
        #     print("Mini-batch Y shape:", batch_images.shape)
        #     # Example:
        #     model_adam = cm.DeepANN.simple_model(image_shape, optimizer='adam')
        #     model_sgd = cm.DeepANN.simple_model(image_shape, optimizer='sgd')
        #     model_rmsprop = cm.DeepANN.simple_model(image_shape, optimizer='rmsprop')
        #     model_adamax = cm.DeepANN.simple_model(image_shape, optimizer='adamax')
        #     cm.compare_model([model_adam, model_sgd, model_rmsprop, model_adamax], train_generator, validation_generator, epochs=20)
        print("rnn")



    def option4():
        my_model = cm.DeepANN()  # Change to regular.DeepANN()
        m1 = my_model.vgg_model()
        Ann_history = m1.fit(train_generator, epochs=5, validation_data=validation_generator)

        # Plotting training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(Ann_history.history['loss'], label='Training Loss')
        plt.plot(Ann_history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plotting training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(Ann_history.history['accuracy'], label='Training Accuracy')
        plt.plot(Ann_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        Ann_test_loss, Ann_test_acc = m1.evaluate(test_generator)
        print(f'Test Accuracy: {Ann_test_acc}')
        m1.save('CNN_VGG_Model.keras')
        print(m1.summary())


    def option5():
        input_shape = (28, 28, 3)

        lstm_Model = cm.DeepANN()
        m1 = lstm_Model.lstm_model(input_shape)
        lstm_history = m1.fit(train_generator, epochs=5, validation_data=validation_generator)

        # Plotting training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(lstm_history.history['accuracy'], label='Training Accuracy')
        plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        LSTM_test_loss, LSTM_test_acc = m1.evaluate(train_generator)
        print(f'Test Accuracy: {LSTM_test_acc}')
        m1.save('LSTMModel.keras')
        print(m1.summary())

    def option6():
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=(28, 28),
            batch_size=32,
            class_mode='input'
        )

        test_generator = test_datagen.flow_from_directory(
            'train',
            target_size=(28, 28),
            batch_size=32,
            class_mode='input'
        )

        # Model architecture
        input_img = Input(shape=(28, 28, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Model training
        history = autoencoder.fit(train_generator, epochs=5, validation_data=test_generator)

        # Evaluation
        loss = autoencoder.evaluate(test_generator)
        print("Test loss:", loss)

        # Visualizing original and reconstructed images
        num_samples = 5
        sample_images = next(test_generator)[0][:num_samples]
        reconstructed_images = autoencoder.predict(sample_images)

        # Function to display original and reconstructed images
        def display_images(original_images, reconstructed_images):
            plt.figure(figsize=(10, 4 * num_samples))
            for i in range(num_samples):
                # Original images
                plt.subplot(num_samples, 2, 2 * i + 1)
                plt.imshow(original_images[i])
                plt.title("Original")
                plt.axis('off')

                # Reconstructed images
                plt.subplot(num_samples, 2, 2 * i + 2)
                plt.imshow(reconstructed_images[i])
                plt.title("Reconstructed")
                plt.axis('off')
            plt.show()

        # Display the original and reconstructed images
        display_images(sample_images, reconstructed_images)
    def exit_program():
        print("Exiting program")
        quit()


    # Menu options
    print("===========================================================================================")
    menu = {
        '1': option1,
        '2': option2,
        '3': option3,
        '4': option4,
        '5': option5,
        '6': option6,
        '7': exit_program
    }

    while True:
        # Display menu
        print("\nMenu:")
        print("1. ANN Model")
        print("2. CNN Model")
        print("3. Mini-Batch")
        print("4. VGG Model")
        print("5. LSTM")
        print("6. AutoEncoder")
        print("7. Exit")


        choice = input("Enter your choice: ")

        if choice in menu:
            menu[choice]()
        else:
            print("Invalid choice. Please try again.")
