# Image Classifier Written with MXNet + Flask

<p align="center">
<img src="https://raw.githubusercontent.com/XD-DENG/flask-app-for-mxnet-img-classifier/master/static/img/screenshot.png" alt="Drawing" style="width:40%;"/>
</p>


A Flask (Python) Web Interface for [MXNet](http://mxnet.io/) Image Classifier.

This app simply invoked the [pre-trained model](http://mxnet.io/tutorials/python/predict_image.html) provided by MXNet community. 


## Deployment Using Docker

```bash
docker run -p 8000:8000 xddeng/flask-app-for-mxnet-img-classifier:v2
```

Now you can try to access the service at http://localhost:8000


## Deployment

### Step - 1: Environment
```bash
sudo yum install python-pip
sudo yum install git
sudo yum install numpy opencv*

pip install Flask
pip install mxnet
pip install gunicorn
```

### Step - 2: Clone This Project

```bash
git clone https://github.com/XD-DENG/flask-app-for-mxnet-img-classifier.git
```

### Step - 3: Download Pre-Trained MXNet Model

From http://data.mxnet.io/models/imagenet-11k/, download

- resnet-152/resnet-152-symbol.json
- resnet-152/resnet-152-0000.params
- synset.txt

Note that we need to put all these three files under application directory.

### Step - 4: Start Service

```bash
gunicorn -b 0.0.0.0:80 app:app
```
