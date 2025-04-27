# AI/main.py

import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
import requests
import tensorflow as tf

# 0) تهيئة المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) تهيئة Firestore
service_account_path = os.path.join(BASE_DIR, "serviceAccount.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
db = firestore.Client()

# 2) تحميل نموذج TFLite
model_path = os.path.join(BASE_DIR, "model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3) تعريف موديلات البيانات
class ItemIn(BaseModel):
    id: str
    imageUrl: str

class MatchOut(BaseModel):
    itemId: str
    matchItemId: str
    confidence: float

# 4) إنشاء تطبيق FastAPI
app = FastAPI()

def fetch_and_preprocess(url, size=(224, 224)):
    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
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
    # استرجاع العناصر من Firestore
    docs = db.collection("items").stream()
    found = []
    for doc in docs:
        data = doc.to_dict()
        found.append(ItemIn(id=doc.id, imageUrl=data.get("imageUrl", "")))

    results = []
    for lost in payload:
        lost_img = fetch_and_preprocess(lost.imageUrl)
        for f in found:
            if f.id == lost.id:
                continue
            try:
                found_img = fetch_and_preprocess(f.imageUrl)
                score = infer(lost_img - found_img)
                results.append(MatchOut(
                    itemId=lost.id,
                    matchItemId=f.id,
                    confidence=score
                ))
            except Exception as e:
                # تجاهل الصور الفاسدة أو المشاكل أثناء التحميل
                continue
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:5]
