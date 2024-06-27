import matplotlib.pyplot as plt
from keras import *
from keras.layers import *
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def train_model(model_instance, train_generator, validate_generator, epochs=5):
    mhistory = model_instance.fit(train_generator, validation_data=validate_generator, epochs=epochs)
    return mhistory

def compare_model(models,train_generator,validate_generator,epochs=5):
    histories=[]
    for model in models:
        history=train_model(model,train_generator,validate_generator,epochs=epochs)
        histories.append(history)
    plt.figure(figsize=(10,6))
    for i,history in enumerate(histories):
        plt.plot(history.history['accuracy'],label=f'Model{i+1}')
    plt.title('model accuracy comparision')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show(block=True)


class DeepANN():
    def simple_model(self,input_shape=(28,28,3),optimizer='sgd'):
        model=Sequential()
        input_shape=input_shape
        #model.add(Flatten())
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

    def vgg_model(self):
        try:
            model = Sequential()

            # First block
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 3)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))

            # Second block
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))

            # Third block
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D((2, 2)))

            # Flatten and fully connected layers
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(2, activation='softmax'))  # Assuming binary classification

            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            return model

        except Exception as e:
            print("An error occurred:", e)

    def lstm_model(self, input_shape):
        model = Sequential()
        model.add(Reshape((input_shape[0] * input_shape[1], input_shape[2]), input_shape=input_shape))
        model.add(LSTM(units=64, activation='relu'))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


class CNN():
      def simple_cnn():
            model=models.Sequential([
            layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64,(3,3),activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64,activation='relu'),
            layers.Dense(2,activation='softmax')
                ])


        
            model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
            return model

      def cnn_batch():
          model = models.Sequential([
              layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
              layers.BatchNormalization(),  # Adding BatchNormalization layer
              layers.MaxPooling2D((2, 2)),
              layers.Conv2D(64, (3, 3), activation='relu'),
              layers.BatchNormalization(),  # Adding BatchNormalization layer
              layers.MaxPooling2D((2, 2)),
              layers.Conv2D(64, (3, 3), activation='relu'),
              layers.BatchNormalization(),  # Adding BatchNormalization layer
              layers.Flatten(),
              layers.Dense(64, activation='relu'),
              layers.Dense(2, activation='softmax')
          ])
          model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
          return model





3