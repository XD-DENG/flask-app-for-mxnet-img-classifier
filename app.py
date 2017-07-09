from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
import mxnet as mx
import  hashlib
import datetime

Batch = namedtuple('Batch', ['data'])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # restrict the size of the file uploaded

# Prapare the MXNet model (pre-trained)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]



def get_image(url, local=False):
    # download and show the image
    if local == True:
        fname = url
    else:
        fname = mx.test_utils.download(url, dirname="static/img_pool")
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    #os.remove(fname)

    if img is None:
         return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def mx_predict(url, local=False):
    img = get_image(url, local)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    result = []
    for i in a[0:5]:
        result.append((labels[i].split(" ", 1)[1], round(prob[i], 3)))
    return result




@app.route("/", methods = ['POST', "GET"])
def FUN_root():
    if request.method == "POST":
        img_url = request.form.get("img_url")
        prediction_result = mx_predict(img_url)
        print prediction_result
        return render_template("index.html", img_src = img_url, prediction_result = prediction_result)
    else:
        return render_template("index.html")



@app.route("/about/")
def FUN_about():
    return render_template("about.html")



ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_image", methods = ['POST'])
def FUN_upload_image():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return(redirect(url_for("FUN_root")))
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            return(redirect(url_for("FUN_root")))
        if file and allowed_file(file.filename):
            filename = os.path.join("static/img_pool", hashlib.sha256(str(datetime.datetime.now())).hexdigest() + secure_filename(file.filename).lower())
            file.save(filename)
            prediction_result = mx_predict(filename, local=True)
            return render_template("index.html", img_src = filename, prediction_result = prediction_result)
    return(redirect(url_for("FUN_root")))



if __name__ == "__main__":
    app.run(debug=True)
