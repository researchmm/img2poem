#-*-coding:utf-8-*-
from flask import Flask, request, render_template, jsonify
import os
import sys
import numpy as np
import extract_feature
import generate_poem

def get_poem(image_file):
    img_feature = extract_feature.get_feature(image_file)
    return generate_poem.generate(img_feature)

app = Flask(__name__)
app.debug = False 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/reco', methods=['POST'])
def reco():
    files = request.files
    f = files['file']
    filename = f.filename
    filetype = filename.split('.')[-1].lower()
    if filetype != 'jpg' and filetype != 'png' and filetype != 'jpeg':
        return jsonify(result = "")
    f.save('../images/tmp/' + filename)
    imagePath = '../images/tmp/' + filename
    result = get_poem(imagePath)[0].replace('\n', '<br>') 
    return jsonify(result = result)

if __name__ == '__main__':
    app.run("0.0.0.0", 8086)
