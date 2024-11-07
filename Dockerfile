FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
COPY devel-requirements.txt /tmp/devel-requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && pip install -r /tmp/devel-requirements.txt \
    && rm /tmp/requirements.txt /tmp/devel-requirements.txt

COPY pcfnet/ /tmp/pcfnet/pcfnet/
COPY setup.py /tmp/pcfnet/setup.py
COPY README.md /tmp/pcfnet/README.md
COPY LICENSE /tmp/pcfnet/LICENSE
COPY MANIFEST.in /tmp/pcfnet/MANIFEST.in
RUN pip install /tmp/pcfnet[all] \
    && rm -r /tmp/pcfnet

CMD ["/bin/bash"]