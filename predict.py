import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

class LogoClassification:
    def __init__(self,filename):
        self.filename =filename


    def getPrediction(self):

        model_path = "models/modelnew.h5"
        loaded_model = tf.keras.models.load_model(model_path)

        imagename = self.filename
        image = cv2.imread(imagename)

        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((200,200))
        expand_input = np.expand_dims(resize_image,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255
        preds = loaded_model.predict(input_data)
        result = preds.argmax()
        return repr(result)
        # if result[0]['Dawn'] == 0:
        #     prediction = 'knns'
        #     return [{"image": prediction}]
        # elif result == 1:
        #     prediction = 'pkmeat'
        #     return [{"image": prediction}]
        # elif result == 2:
        #     prediction = 'McCain'
        #     return [{"image": prediction}]
        # elif result == 3:
        #     prediction = 'Dawn'
        #     return [{"image": prediction}]
        # else:
        #     return [{"ERROR": "Please select another image. !!!"}]


        
