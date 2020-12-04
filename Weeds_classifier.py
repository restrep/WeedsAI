from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from keras.applications.resnet50 import preprocess_input


#############################
# Global variables
#############################

ALLOWED_EXTENSION  =set(['pdf', 'png','jpg','jpeg'])
IMAGE_HEIGHT =224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSION


#############################
# Flask APP
#############################

app = Flask(__name__)

model = load_model('model.h5')
num2class = {0: 'Chinee apple', 1: 'Lantana', 2: 'Negative', 3: 'Parkinsonia', 4: 'Parthenium', 5: 'Prickly acacia', 6: 'Rubber vine', 7: 'Siam weed', 8: 'Snake weed'}

@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']
    
    if file.filename =='':
        return render_template('ImageML.html', prediction = 'You did not select an image')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) # remove weird characters form filename
        print("***"+filename)
        x = []
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        img  = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        x  = np.array(img)
        x = np.expand_dims(x, axis=0)
        x  = preprocess_input(x)
        pred = model.predict(x)
        sort = np.argsort(-pred)
        label1 = sort[0][0]
        label2 = sort[0][1]
        prob1 = round(pred[0][label1] * 100, 1)
        prob2 =  round(pred[0][label2] * 100, 1)
        name1 = num2class[label1]
        name2 = num2class[label2] 
        
        return render_template('Pred.html', prediction = f'The image is predicted as {name1} weed with {prob1}% probability and as {name2} with {prob2}% probability. \n')
    else:
        return render_template('Pred.html', prediction = 'Invalid File extension. \n')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
            
