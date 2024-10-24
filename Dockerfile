FROM nvcr.io/nvidia/tritonserver:22.12-py3

COPY ./models /models

RUN git clone https://github.com/triton-inference-server/python_backend -b r22.12
RUN pip install transformers==4.32.1
RUN pip install optimum
RUN pip install onnxruntime onnx
RUN export PYTHONIOENCODING=UTF-8

CMD ["tritonserver", "--model-repository=/models"] 
