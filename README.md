# flask-app-for-mxnet-img-classifier

[**Demo**](http://side-1.seekingqed.com)

A Flask (Python) Web Interface for [MXNet](http://mxnet.io/) Image Classifier.

This app simply invoked the [pre-trained model](http://mxnet.io/tutorials/python/predict_image.html) provided by MXNet community. 

## Deployment

### Step - 1: Environment
```bash
sudo yum install python-pip
sudo yum install git

sudo pip install Flask
sudo yum install numpy opencv*
sudo pip install mxnet

sudo pip install gunicorn
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
sudo gunicorn -b 0.0.0.0:80 app:app
```
