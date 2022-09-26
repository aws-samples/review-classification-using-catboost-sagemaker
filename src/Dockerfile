FROM nvidia/cuda:11.7.0-base-centos7
CMD nvidia-smi

# Install the following:
#1. Header files and static libraries for python dev
#2. Required dependencies
#3. Setup Deep Learning/Machine Learning libraries.
# Note: Multiple statements are being combined into one RUN command to minimize the number of layers Docker creates
#that translates into smaller docker image size

RUN yum -y install epel-release && yum -y update && yum install wget -y
RUN yum -y groupinstall "Development Tools" && yum -y install openssl-devel bzip2-devel libffi-devel xz-devel
RUN gcc --version
RUN wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
RUN tar xvf Python-3.8.12.tgz -C /usr/bin/
RUN pwd

WORKDIR /usr/bin/Python-3.8.12/

RUN ./configure --enable-optimizations
RUN make altinstall
RUN python3.8 --version
RUN pip3.8 --version

WORKDIR /

RUN yum update -y && yum install -y wheel python3-devel python3-libs python3-tools python3-pip && \
    yum install gcc tar make util-linux kmod man sudo git -y && \
    yum install wget -y && yum install -y unzip && yum install gcc-c++ -y && yum clean all \
    && yum install nginx -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
&& unzip awscliv2.zip \
&& sudo ./aws/install \
&& /usr/local/bin/aws --version

#RUN docker run --mount type=bind,source="~/.aws",target=/root/.aws
RUN pip3.8 install --no-cache-dir --upgrade pip && \pip3.8 install --no-cache-dir --upgrade setuptools && \
pip3.8 install --no-cache-dir numpy==1.21.0 Cython

RUN pip3.8 install --no-cache-dir scipy==1.7.3 scikit-learn==1.0.2 pandas==1.4.0 Pillow==9.0.0 \
flask gevent gunicorn boto3 s3fs joblib catboost==1.0.0

##########################################################################
# Download and install CUDA. Sagemaker requires CUDA and CuDNN environments built in the docker image. It will only
# mount the NVIDIA driver from the host machine.

RUN pip3.8 install shap==0.40.0

# Create the directories required by Sagemaker. /opt/ml and all sub-directories are reserved by Amazon SageMaker training.
RUN mkdir /opt/ml /opt/ml/input /opt/ml/input/config /opt/ml/input/data /opt/ml/input/data/training /opt/ml/model /opt/ml/output /opt/program

# Set PYTHONPATH env. This is important for python to import python modules from codes present in sub-directories.
ENV PYTHONPATH=/opt/program
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# See https://github.com/NVIDIA/nvidia-docker/issues/1409
ENV NVIDIA_DISABLE_REQUIRE=1

# Update the Python version in PATH to python3
RUN rm -f /usr/bin/python && rm -f /usr/bin/python3 && ln -s /usr/bin/Python-3.8.12/python /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

#  Set up the program in the image
COPY code/* /opt/program/
WORKDIR /opt/program/

