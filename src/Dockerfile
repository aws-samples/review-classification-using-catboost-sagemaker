 FROM amazonlinux:1

# Install the following:
#1. Header files and static libraries for python dev
#2. Required dependencies
#3. Setup Deep Learning/Machine Learning libraries.
# Note: Multiple statements are being combined into one RUN command to minimize the number of layers Docker creates
#that translates into smaller docker image size

RUN yum update -y && yum install -y python36 python36-devel python36-libs python36-tools python36-pip && \
yum install gcc tar make wget util-linux kmod man sudo git -y && \
yum install wget -y && \
yum install aws-cli -y && \
yum install nginx -y && \
yum install gcc-c++.noarch -y && yum clean all

RUN pip-3.6 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --upgrade setuptools && \
    pip3 install Cython && \
    pip3 install --no-cache-dir numpy==1.16.0 scipy==1.4.1 scikit-learn==0.20.3 pandas==0.24.2 flask gevent gunicorn boto3 s3fs  matplotlib joblib catboost==0.20.2


# Set LD_LIBRARY_PATH to OS paths. This makes sure that we pick OS shared libraries when in conflict with Brazil shared libraries
ENV LD_LIBRARY_PATH=/lib:/lib64:/usr/lib64:/usr/lib

CMD python -m pip3 install git+https://github.com/Theano/Theano.git@rel-1.0.0#egg=Theano

# RUN wget https://cmake.org/files/v3.9/cmake-3.9.1-Linux-x86_64.tar.gz \
RUN wget https://cmake.org/files/v3.17/cmake-3.17.3-Linux-x86_64.tar.gz \
&& tar xvzf cmake-3.17.3-Linux-x86_64.tar.gz \
&& rm cmake-3.17.3-Linux-x86_64.tar.gz
RUN git clone -b v0.7.1 https://github.com/Theano/libgpuarray.git
WORKDIR "/libgpuarray"
RUN git checkout tags/v0.7.1 -b v0.7.1
RUN mkdir Build
WORKDIR "Build"

RUN /cmake-3.17.3-Linux-x86_64/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
&& make \
&& make install
WORKDIR "/libgpuarray"




RUN python3 setup.py build_ext -L /usr/lib -I /usr/include \
&& python3 setup.py build \
&& python3 setup.py install

# Setup CUDNN & CUDA Paths, Theano Flags, and Backend to Theano
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
                  CPATH=/usr/local/cuda/include:$CPATH \
                  LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH \
                  PATH=$PATH:/usr/local/cuda/bin:$PATH \
                  THEANO_FLAGS="device=cuda,floatX=float32" \
                  KERAS_BACKEND=theano

##########################################################################
# Download and install CUDA. Sagemaker requires CUDA and CuDNN environments built in the docker image. It will only
# mount the NVIDIA driver from the host machine.

WORKDIR /
RUN mkdir data
WORKDIR data

RUN wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run \
&& chmod u+x cuda_9.0.176_384.81_linux-run \
&& ./cuda_9.0.176_384.81_linux-run --tmpdir=/data --silent --toolkit --override \
&& wget <virtual-hosted–style access path to the driver>
#&& wget https://custom-gpu-sagemaker-image.s3.amazonaws.com/installation/cudnn-9.0-linux-x64-v7.tgz \
&& tar -xvzf cudnn-9.0-linux-x64-v7.tgz \
&& cp /data/cuda/include/cudnn.h /usr/local/cuda/include \
&& cp /data/cuda/lib64/libcudnn* /usr/local/cuda/lib64 \
&& chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* \
&&  rm -rf /data/*


# Create the directories required by Sagemaker. /opt/ml and all sub-directories are reserved by Amazon SageMaker training.
RUN mkdir /opt/ml /opt/ml/input /opt/ml/input/config /opt/ml/input/data /opt/ml/input/data/training /opt/ml/model /opt/ml/output /opt/program

# Set PYTHONPATH env. This is important for python to import python modules from codes present in sub-directories.

ENV PYTHONPATH=/opt/program
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set NVIDIA mount environments
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"


#  Set up the program in the image
COPY code/* /opt/program/
WORKDIR /opt/program

