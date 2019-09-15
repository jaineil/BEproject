from flask import Flask, request, render_template, jsonify, Response
import json
import sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import os

app = Flask(__name__)


@app.route('/')
def index():
    return Response(open('index.html').read(), mimetype='text/html')


@app.route('/process', methods=['POST'])
def process():
    f = open("demofile.py", "w")
    nb = new_notebook()
    json_string = request.form['state']
    datastore = json.loads(json_string)
    print(datastore)
    optimizer = str(datastore["optimizer"])
    content = '''
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),'''
    
    for i in range(datastore["layers"]["count"]):
        inputsize = list(map(int, datastore["layers"][str(i+1)]['inputsize'].rstrip().split()))
        outputsize = list(map(int, datastore["layers"][str(i+1)]['outputsize'].rstrip().split()))
        if len(inputsize) == 0:
            content = content + '''
        tf.keras.layers.Dense({}, activation=tf.nn.{}),'''.format(outputsize, datastore["layers"][str(i+1)]["act"])
        else:
            content = content + '''
        tf.keras.layers.Dense({}, input_shape={}, activation=tf.nn.{}),'''.format(outputsize, 
        inputsize,
        datastore["layers"][str(i+1)]["act"])
    
    content = content + '''])
model.compile(optimizer='{}',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])'''.format(optimizer)
    f.write(content)
    nb.cells.append(new_code_cell(content))
    nbformat.write(nb, 'demofile.ipynb')
    os.system('kernel-run demofile.ipynb')
    f.close()


if __name__ == "__main__":
    app.run(debug=True)
