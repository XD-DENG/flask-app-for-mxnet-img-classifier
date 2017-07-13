from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from collections import namedtuple
import mxnet as mx
import  hashlib
import datetime



app = Flask(__name__)
# restrict the size of the file uploaded
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


################################################
# Error Handling
################################################

@app.errorhandler(404)
def FUN_404(error):
    return render_template("error.html")

@app.errorhandler(405)
def FUN_405(error):
    return render_template("error.html")

@app.errorhandler(500)
def FUN_500(error):
    return render_template("error.html")


################################################
# Functions for running classifier
################################################

# define a simple data batch
Batch = namedtuple('Batch', ['data'])

# Prapare the MXNet model (pre-trained)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


def get_image(file_location, local=False):
    # users can either 
    # [1] upload a picture (local = True)
    # or
    # [2] provide the image URL (local = False)
    if local == True:
        fname = file_location
    else:
        fname = mx.test_utils.download(file_location, dirname="static/img_pool")
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    if img is None:
         return None
    
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]

    return img


def mx_predict(file_location, local=False):
    img = get_image(file_location, local)

    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()

    # Return the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    result = []
    for i in a[0:5]:
        result.append((labels[i].split(" ", 1)[1], round(prob[i], 3)))

    return result


################################################
# Functions for Image Archive
################################################

def FUN_resize_img(filename, resize_proportion = 0.3):
    '''
    FUN_resize_img() will resize the image passed to it as argument to be {resize_proportion} of the original size.
    '''
    img=cv2.imread(filename)
    small_img = cv2.resize(img, (0,0), fx=resize_proportion, fy=resize_proportion)
    cv2.imwrite(filename, small_img)

################################################
# Functions Building Endpoints
################################################

@app.route("/", methods = ['POST', "GET"])
def FUN_root():
	# Run correspoing code when the user provides the image url
	# If user chooses to upload an image instead, endpoint "/upload_image" will be invoked
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
	    FUN_resize_img(filename)
            return render_template("index.html", img_src = filename, prediction_result = prediction_result)
    return(redirect(url_for("FUN_root")))


################################################
# Start the service
################################################
if __name__ == "__main__":
    app.run(debug=True)
