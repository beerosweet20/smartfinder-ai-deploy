import os
import json
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
from google.oauth2 import service_account
import requests
import tensorflow as tf

# تحقق من وجود متغير البيئة PRIVATE_KEY
private_key = os.getenv("PRIVATE_KEY")
if private_key is not None:
    private_key = private_key.replace("\\n", "\n")
else:
    raise ValueError("متغير البيئة PRIVATE_KEY غير مضبوط.")

# إعداد اتصال فايربيس من متغيرات البيئة
service_account_info = {
    "type": os.getenv("TYPE"),
    "project_id": os.getenv("PROJECT_ID"),
    "private_key_id": os.getenv("PRIVATE_KEY_ID"),
    "private_key": private_key,  # استخدم هنا المتغير المعدل
    "client_email": os.getenv("CLIENT_EMAIL"),
    "client_id": os.getenv("CLIENT_ID"),
    "auth_uri": os.getenv("AUTH_URI"),
    "token_uri": os.getenv("TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
}

credentials = service_account.Credentials.from_service_account_info(service_account_info)
db = firestore.Client(credentials=credentials, project=service_account_info["project_id"])

# تحميل نموذج TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# موديلات الطلب والاستجابة
class ItemIn(BaseModel):
    id: str
    imageUrl: str

class MatchOut(BaseModel):
    itemId: str
    matchItemId: str
    confidence: float

app = FastAPI()

def fetch_and_preprocess(url, size=(224, 224)):
    r = requests.get(url)
    r.raise_for_status()
    arr = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def infer(image_np):
    interpreter.set_tensor(input_details[0]['index'], image_np)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output.squeeze())

@app.post("/match", response_model=list[MatchOut])
def match_endpoint(payload: list[ItemIn]):
    docs = db.collection("items").stream()
    found = []
    for doc in docs:
        data = doc.to_dict()
        found.append(ItemIn(id=doc.id, imageUrl=data["imageUrl"]))

    results = []
    for lost in payload:
        lost_img = fetch_and_preprocess(lost.imageUrl)
        for f in found:
            if f.id == lost.id:
                continue
            score = infer(lost_img - fetch_and_preprocess(f.imageUrl))
            results.append(MatchOut(
                itemId=lost.id,
                matchItemId=f.id,
                confidence=score
            ))

    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:5]
