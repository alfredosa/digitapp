from flask import Flask, render_template
from flask import request
import base64
from PIL import Image
from numpy import asarray
import tensorflow as tf

app = Flask(__name__)

# load the image
def processAndPredict():
    image = Image.open('app/static/img/imageformodel.png').convert('P')
    image = image.resize((252,252),Image.ANTIALIAS)
    image = image.resize((28,28),Image.ANTIALIAS)
    # convert image to numpy array
    data = (asarray(image)/8).reshape(1,28,28,1)

    model = tf.keras.models.load_model('app/static/saved_model/my_model')
    prediction = model.predict_classes(tf.constant(data))
    print(prediction)
    return prediction

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def data():
    if request.method == 'POST':
        form_text = request.form.get("URL")
        new_data = form_text.replace('data:image/png;base64,', '')
        imgdata = base64.b64decode(new_data)
        filename = 'app/static/img/imageformodel.png'
        with open(filename, 'wb') as f:
                f.write(imgdata)
        prediction = processAndPredict()
        text = f'<img id="canvasimg" src="{form_text}"><p style="font-size: 50px"><strong>Predicted Value: {prediction[0]}</strong></p>'
        return render_template("index.html", predictionval = prediction[0])
