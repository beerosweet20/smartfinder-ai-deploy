# AI/git/main.py

import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
import requests
import tensorflow as tf

# 1) تهيئة Firestore بطريقة ذكية
service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccount.json")
db = firestore.Client.from_service_account_json(service_account_path)

# 2) تحميل نموذج TFLite
model_path = os.getenv("MODEL_PATH", "model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3) موديلات البيانات
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
