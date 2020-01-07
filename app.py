from flask import Flask, request, render_template, jsonify, Response
import json
import sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import os
import time 

app = Flask(__name__)


@app.route('/')
def index():
    return Response(open('index.html').read(), mimetype='text/html')

@app.route('/fileupload', methods=['POST'])
def fileupload():
    if 'file' in request.files:  
        f = request.files['file']  
        f.save(f.filename)
    return ('',204)

@app.route('/process', methods=['POST'])
def process():
    f = open("demofile.py", "w")
    nb = new_notebook()
    json_string = request.form['state']
    datastore = json.loads(json_string)
    dataset = datastore['dataset']
    activations = ["Dropout", "MaxPooling"]
    optimizer = str(datastore["optimizer"])
    content = '''
import tensorflow as tf
dataset = tf.keras.datasets.{}
(x_train, y_train), (x_test, y_test) = dataset.load_data()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=({}))'''.format(dataset, datastore["layers"]["1"]["inputsize"])
    
    for i in range(2, datastore["layers"]["count"]+1):
        if datastore["layers"][str(i)]["type"] in activations:
            content = content + '''
        ,tf.keras.layers.Dropout({})'''.format(datastore["layers"][str(i)]["outputsize"])
        else:
            content = content + '''
        ,tf.keras.layers.Dense({}, activation=tf.nn.{})'''.format(int(datastore["layers"][str(i)]['outputsize']), datastore["layers"][str(i)]["act"])
    
    content = content + '''])
model.compile(optimizer='{}',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)'''.format(optimizer)
    f.write(content)
    nb.cells.append(new_code_cell(content))
    nbformat.write(nb, 'demofile.ipynb')
    os.system('kaggle kernels push')
    time.sleep(60)
    os.system('kaggle kernels output demofile')
    f.close()
    return ('',204)

@app.route('/testinput', methods=['POST'])
def giveinput():
    runfilecontent = '''
import tensorflow as tf
trained_model = tf.keras.models.load_model('returnfile.h5')

    '''

if __name__ == "__main__":
    app.run(debug=True)
