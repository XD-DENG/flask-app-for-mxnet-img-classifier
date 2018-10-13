FROM centos:7

WORKDIR /app

COPY . /app

# Install pip
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
RUN python get-pip.py

RUN yum -y install gcc
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN yum -y install opencv*

RUN curl "http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json" -o "resnet-152-symbol.json"
RUN curl "http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params" -o "resnet-152-0000.params"
RUN curl "http://data.mxnet.io/models/imagenet-11k/synset.txt" -o "synset.txt"

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]