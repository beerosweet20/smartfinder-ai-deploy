# AI/src/main.py
import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
import requests
import tensorflow as tf

# 1) تهيئة Firestore
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"F:\smartfinder\AI\serviceAccount.json"
db = firestore.Client()

# 2) تحميل نموذج TFLite
interpreter = tf.lite.Interpreter(
    model_path=r"F:\smartfinder\AI\models\model.tflite"
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3) موديل للـ Request
class ItemIn(BaseModel):
    id: str
    imageUrl: str

class MatchOut(BaseModel):
    itemId: str
    matchItemId: str
    confidence: float

app = FastAPI()

def fetch_and_preprocess(url, size=(224,224)):
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
    # 1) استرجاع كل الـ items من Firestore
    docs = db.collection("items").stream()
    found = []
    for doc in docs:
        data = doc.to_dict()
        found.append(ItemIn(id=doc.id, imageUrl=data["imageUrl"]))

    # 2) للمقتنيات المُرسَلة في payload، نحسب التشابه مع كل عنصر في found
    results = []
    for lost in payload:
        lost_img = fetch_and_preprocess(lost.imageUrl)
        for f in found:
            if f.id == lost.id: continue
            score = infer(lost_img - fetch_and_preprocess(f.imageUrl))
            results.append(MatchOut(
                itemId=lost.id,
                matchItemId=f.id,
                confidence=score
            ))
    # 3) نرتب حسب الثقة ونعيد أفضل 5
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results[:5]
