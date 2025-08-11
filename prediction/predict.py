import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class DogBreed:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogbreed(self):
        # load model
        # model = load_model(os.path.join("model", "model_vgg16.h5"))
        model = load_model(os.path.join("Models", "model_vgg16.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        # test_image = image.load_img(imagename, target_size = (64,64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        print("Input shape to model:", test_image.shape)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        class_names = sorted(os.listdir("Dataset/train"))  # adjust path if needed
        pred_index = result[0]
        prediction = class_names[pred_index]
        return [{"image": prediction}]


