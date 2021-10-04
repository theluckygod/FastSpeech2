FROM nvcr.io/nvidia/pytorch:21.07-py3
USER root

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    pip install --upgrade setuptools pip && \
    pip install underthesea && \
    pip install eng-to-ipa && \
    pip install git+git://github.com/theluckygod/prosodic.git && \
    pip install vinorm && \
    pip install viphoneme && \
    pip install llvmlite --ignore-installed && \
    python3 -m pip install -r /tmp/requirements.txt

COPY . /Fastspeech2
WORKDIR /Fastspeech2

CMD /bin/bash