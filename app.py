# -----------------------------------------LIBRARIRES------------------------------------------------
from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from flask import Response
from PIL import Image, ImageFont, ImageDraw
# Importing the Keras libraries and package
# from keras.models import model_from_json
batch_size = 32
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# ----------------------------------------------CNN LAYERS--------------------------------------------
model = Sequential()

model.add(Conv2D(16, (3,3), input_shape= (200,200,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 11, activation = 'softmax'))

model.compile(optimizer = RMSprop(lr=0.001), loss = 'categorical_crossentropy', metrics = ['acc'])

# -------------------------------------LOAD SAVED MODEL----------------------------------------------
model.load_weights('models/signLanguage.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

# --------------------------------------------MAIN WEBSITE PAGE-------------------------

def gen_frames():  
    while True:
       
        
            break
        
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/home', methods=['POST'])
def home():
    global COUNT

    img = request.files['image']
    img.save('static/{}.jpg'.format(COUNT))
    test_image = image.load_img('static/{}.jpg'.format(COUNT), target_size = (200,200))    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    pred = model.predict(test_image)
    COUNT+=1
    # # ------------------------------------JUMP TO PREDICTION PAGE--------------------
    return render_template('prediction.html', data = pred )

@app.route('/')
def man():
    return render_template('index.html')



@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static/', "{}.jpg".format(COUNT-1))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)