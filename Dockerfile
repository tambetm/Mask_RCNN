FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y git wget python-tk

RUN pip install keras cython
RUN pip install 'git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'

RUN git clone https://github.com/tambetm/Mask_RCNN.git
WORKDIR Mask_RCNN
RUN wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

RUN apt-get install -y python3-tk
RUN pip install scikit-image
RUN pip install flask

#ENTRYPOINT ["python3","detect.py"]
ENTRYPOINT ["python3","server.py"]
EXPOSE 5000
