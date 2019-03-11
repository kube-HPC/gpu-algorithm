FROM tensorflow/tensorflow:1.13.1-gpu-py3
LABEL maintainer="yehiyam@gmail.com"

COPY requirements.txt /hkube/algorithm-runner/requirements.txt
RUN pip3 install -r /hkube/algorithm-runner/requirements.txt

COPY . /hkube/algorithm-runner
RUN cd /hkube/algorithm-runner
WORKDIR /hkube/algorithm-runner

CMD ["python3","-u", "app.py"]
