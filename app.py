from flask import Flask, request, render_template, jsonify, Response
import json

app = Flask(__name__)

@app.route('/')
def index():
    return Response(open('index.html').read(),mimetype='text/html')


@app.route('/process', methods=['POST'])
def process():
    f = open("demofile.py","w")
    json_string = request.form['state']
    datastore = json.loads(json_string)
    optimizer = str(datastore["optimizer"])
    content ='''
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),'''
    for i in range(datastore["layers"]["count"]):
        content = content + '''
        tf.keras.layers.Dense(128, activation=tf.nn.{}),'''.format(datastore["layers"][str(i+1)]["act"])
    content = content + '''])
model.compile(optimizer='{}',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])'''.format(optimizer)
    f.write(content)
    f.close()

if __name__=="__main__":
    app.run(debug=True)