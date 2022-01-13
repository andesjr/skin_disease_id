from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
from pydantic import BaseModel
import tensorflow as tf

class Image(BaseModel):
    image: str
    height: int
    width: int
    channel: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def recall(y_true, y_pred):
    y_true = tf.keras.backend.ones_like(y_true)
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    all_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + tf.keras.backend.epsilon())
    return recall

def mean_sensitivity(y_true,y_pred):
    C = 7
    recall_m = recall(y_true,y_pred)
    return recall_m / C

#EDIT the name and/or location of the model
model = tf.keras.models.load_model('skin_cancer_model.h5',custom_objects={'mean_sensitivity': mean_sensitivity,'recall':recall})


@app.get("/")
def home():
    return {"greeting": "Welcome to the skin disease skin project API"}

@app.post("/predict")
def predict_class(Img: Image):
    #decode image
    decoded = base64.b64decode(bytes(Img.image, 'utf-8'))
    decoded = np.frombuffer(decoded, dtype='uint8')
    decoded = decoded.reshape(Img.height, Img.width, Img.channel)[None,:,:,:]
    #predict class

    #EDIT this part to adapt it to your project
    prediction = model.predict(decoded)
    pred_bool = np.argmax(prediction)
    disease_names = ['Mélanome', 'Naevus mélanocytaires', 'Carcinome basocellulaire', 'Kératoses actiniques', 'Lésions bénignes de type kératose', 'Dermatofibrome', 'Lésions vasculaires']
    classes = {0:'Mélanome', 1:'Naevus mélanocytaires', 2:'Carcinome basocellulaire',3:'Kératoses actiniques',4:'Lésions bénignes de type kératose',5:'Dermatofibrome',6:'Lésions vasculaires'}
    dicti = {b:a for a,b in zip(prediction[0], disease_names)}
    dicti_sorted = {k: v for k, v in sorted(dicti.items(), key=lambda item: item[1],reverse=True)}
    #prediction = float(prediction) # the output dtype of the network, np.float32, is not serializable in json
    return {classes[pred_bool] : (prediction.max()*100).round(2)}
