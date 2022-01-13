FROM python:3.8-buster

#tensorflow is installed on upper layer to allow caching this layer in case of multiple build iterations
RUN pip install tensorflow

COPY requirements_docker.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY api api
COPY api/skin_cancer_model.h5 skin_cancer_model.h5

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

COPY /home/andesjr/code/andesjr/gcp/skin-disease-id-4b43f149320f.json /credentials.json
