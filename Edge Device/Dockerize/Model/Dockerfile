FROM python:latest

# Install required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgl1-mesa-dri


# Install required Python packages
RUN apt install -y python3-opencv
RUN pip install opencv-python-headless
RUN pip3 install paho-mqtt
#RUN pip3 install jsonlib
#RUN pip3 install base64
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install tensorflow
#RUN pip3 install time

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

#RUN alias python=python3

# Set entrypoint
ENTRYPOINT [ "python" ]

# Set default command
CMD [ "ML_Model.py" ]

