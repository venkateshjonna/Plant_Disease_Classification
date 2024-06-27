import matplotlib
matplotlib.use('Agg')  # Set Matplotlib backend to 'Agg' to avoid GUI-related warnings


from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, Model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import models

app = Flask(__name__)
model = load_model('cnn_model.h5')
model.summary()
layer_names = []

for layer in model.layers[:8]:
    layer_names.append(layer.name)

print(layer_names)
target_img = os.path.join(os.getcwd(), 'static/images')

# Define a new model that extracts features
feature_extraction_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Allow files with extension png, jpg, and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function to load and prepare the image in the right shape
def read_image(filename):
    img = load_img(filename, target_size=(28, 28))  # Resize the image to (28, 28)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_prediction_and_features(img):
    extracted_features = feature_extraction_model.predict(img)
    class_prediction = model.predict(img)
    return class_prediction, extracted_features


def get_model_summary():
    # Redirect model summary to string
    import sys
    import io
    temp_stdout = sys.stdout
    sys.stdout = io.StringIO()
    model.summary()
    model_summary = sys.stdout.getvalue()
    sys.stdout = temp_stdout  # Reset redirect
    return model_summary




@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path)  # preprocessing method

            # Predict using the preprocessed image directly
            class_prediction, extracted_features = get_prediction_and_features(img)
            classes_x = np.argmax(class_prediction, axis=1)

            # Calculate prediction percentage for each class
            prediction_percentage = [class_prediction[0][0] * 100, class_prediction[0][1] * 100]

            if classes_x[0] == 0:
                plant = "Healthy"

            elif classes_x[0] == 1:
                plant = "Rusty"

            # Print information for debugging
            print("Shape of extracted features:", extracted_features.shape)

            # Get layer activations
            layer_outputs = [layer.output for layer in model.layers[:8]]
            activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
            activations = activation_model.predict(img)

            # Getting Activations of first layer
            first_layer_activation = activations[0]

            # Print shape of first layer activation
            # print("Shape of first layer activation:", first_layer_activation.shape)
            #
            # Visualize activations
            if model =='cnn_model.h5':
                plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis')
                plt.savefig('static/images/first_layer_activation_1.png')
                plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
                plt.savefig('static/images/first_layer_activation_4.png')
                plt.matshow(first_layer_activation[0, :, :, 8], cmap='viridis')
                plt.savefig('static/images/first_layer_activation_8.png')
                plt.matshow(first_layer_activation[0, :, :, 15], cmap='viridis')
                plt.savefig('static/images/first_layer_activation_15.png')
                plt.matshow(first_layer_activation[0, :, :, 21], cmap='viridis')
                plt.savefig('static/images/first_layer_activation_21.png')
                plt.matshow(first_layer_activation[0, :, :, 31], cmap='viridis')
                plt.savefig('static/images/first_layer_activation_31.png')

            # Pass extracted_features to predict.html if valid
            if extracted_features is not None and len(extracted_features.shape) > 1:
                model_summary = get_model_summary()  # Get model summary
                return render_template('predict.html', plant=plant,
                                       user_image=file_path, features=extracted_features, model_summary=model_summary)
            else:
                return "Failed to extract features or the shape is invalid."
        else:
            return "Unable to read the file. Please check the file extension."


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)


