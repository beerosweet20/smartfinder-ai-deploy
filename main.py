# import os
# import json
# import cv2
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from google.cloud import firestore
# from google.oauth2 import service_account
# import requests
# import tensorflow as tf

# # تحقق من وجود متغير البيئة PRIVATE_KEY
# private_key = os.getenv("PRIVATE_KEY")
# if private_key is not None:
#     private_key = private_key.replace("\\n", "\n")
# else:
#     raise ValueError("متغير البيئة PRIVATE_KEY غير مضبوط.")

# # إعداد اتصال فايربيس من متغيرات البيئة
# service_account_info = {
#     "type": os.getenv("TYPE"),
#     "project_id": os.getenv("PROJECT_ID"),
#     "private_key_id": os.getenv("PRIVATE_KEY_ID"),
#     "private_key": private_key,  # استخدم هنا المتغير المعدل
#     "client_email": os.getenv("CLIENT_EMAIL"),
#     "client_id": os.getenv("CLIENT_ID"),
#     "auth_uri": os.getenv("AUTH_URI"),
#     "token_uri": os.getenv("TOKEN_URI"),
#     "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
#     "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
# }

# credentials = service_account.Credentials.from_service_account_info(service_account_info)
# db = firestore.Client(credentials=credentials, project=service_account_info["project_id"])

# # تحميل نموذج TFLite
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # موديلات الطلب والاستجابة
# class ItemIn(BaseModel):
#     id: str
#     imageUrl: str

# class MatchOut(BaseModel):
#     itemId: str
#     matchItemId: str
#     confidence: float

# app = FastAPI()

# def fetch_and_preprocess(url, size=(224, 224)):
#     r = requests.get(url)
#     r.raise_for_status()
#     arr = np.frombuffer(r.content, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, size).astype("float32") / 255.0
#     return np.expand_dims(img, axis=0)

# def infer(image_np):
#     interpreter.set_tensor(input_details[0]['index'], image_np)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])
#     return float(output.squeeze())

# @app.post("/match", response_model=list[MatchOut])
# def match_endpoint(payload: list[ItemIn]):
#     docs = db.collection("items").stream()
#     found = []
#     for doc in docs:
#         data = doc.to_dict()
#         found.append(ItemIn(id=doc.id, imageUrl=data["imageUrl"]))

#     results = []
#     for lost in payload:
#         lost_img = fetch_and_preprocess(lost.imageUrl)
#         for f in found:
#             if f.id == lost.id:
#                 continue
#             score = infer(lost_img - fetch_and_preprocess(f.imageUrl))
#             results.append(MatchOut(
#                 itemId=lost.id,
#                 matchItemId=f.id,
#                 confidence=score
#             ))

#     results.sort(key=lambda x: x.confidence, reverse=True)
#     return results[:5]


# import os
# import cv2
# import json
# import numpy as np
# import requests
# import tensorflow as tf

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from google.cloud import firestore
# from google.oauth2 import service_account

# # 1. اقرأ المفتاح الخاص وحَوِّل "\\n" إلى أسطر فعلية
# private_key = os.getenv("private_key")
# if not private_key:
#     raise ValueError("متغير البيئة private_key غير مضبوط.")
# private_key = private_key.replace("\\n", "\n")

# # 2. جمع كل حقول حساب الخدمة من متغيّرات البيئة
# service_account_info = {
#     "type":                          os.getenv("type"),
#     "project_id":                    os.getenv("project_id"),
#     "private_key_id":                os.getenv("private_key_id"),
#     "private_key":                   private_key,
#     "client_email":                  os.getenv("client_email"),
#     "client_id":                     os.getenv("client_id"),
#     "auth_uri":                      os.getenv("auth_uri"),
#     "token_uri":                     os.getenv("token_uri"),
#     "auth_provider_x509_cert_url":   os.getenv("auth_provider_x509_cert_url"),
#     "client_x509_cert_url":          os.getenv("client_x509_cert_url"),
#     # إذا كنت بحاجة لمتغيّر universe_domain في مكان آخر بالتطبيق،
#     # يمكنك قراءته هنا:
#     # "universe_domain":            os.getenv("universe_domain"),
# }

# # 3. إنشاء Credentials وعميل Firestore
# credentials = service_account.Credentials.from_service_account_info(service_account_info)
# db = firestore.Client(credentials=credentials, project=service_account_info["project_id"])

# # 4. إعداد نموذج TFLite
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # 5. نماذج البيانات
# class ItemIn(BaseModel):
#     id: str
#     imageUrl: str

# class MatchOut(BaseModel):
#     itemId: str
#     matchItemId: str
#     confidence: float

# app = FastAPI()

# # 6. دوال المساعدة
# def fetch_and_preprocess(url: str, size=(224, 224)) -> np.ndarray:
#     r = requests.get(url)
#     r.raise_for_status()
#     arr = np.frombuffer(r.content, np.uint8)
#     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, size).astype("float32") / 255.0
#     return np.expand_dims(img, axis=0)

# def infer(image_np: np.ndarray) -> float:
#     interpreter.set_tensor(input_details[0]['index'], image_np)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details[0]['index'])
#     return float(output.squeeze())

# # 7. نقطة الدخول الرئيسية
# @app.post("/match", response_model=list[MatchOut])
# def match_endpoint(payload: list[ItemIn]):
#     # جلب كل العناصر المسجّلة في Firestore
#     docs = db.collection("items").stream()
#     found = []
#     for doc in docs:
#         data = doc.to_dict()
#         found.append(ItemIn(id=doc.id, imageUrl=data["imageUrl"]))

#     # حساب التشابه لكل عنصر فقدان مقابل الموجودين
#     results = []
#     for lost in payload:
#         lost_img = fetch_and_preprocess(lost.imageUrl)
#         for f in found:
#             if f.id == lost.id:
#                 continue
#             score = infer(lost_img - fetch_and_preprocess(f.imageUrl))
#             results.append(MatchOut(
#                 itemId=lost.id,
#                 matchItemId=f.id,
#                 confidence=score
#             ))

#     # إرجاع أفضل 5 نتائج
#     results.sort(key=lambda x: x.confidence, reverse=True)
#     return results[:5]


import os
import cv2
import json
import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import firestore
from google.oauth2 import service_account
from sklearn.metrics.pairwise import cosine_similarity

# 1) تهيئة Firestore باستخدام متغيرات البيئة
service_account_info = {
    "type": os.getenv("type"),
    "project_id": os.getenv("project_id"),
    "private_key_id": os.getenv("private_key_id"),
    "private_key": os.getenv("private_key").replace("\\n", "\n"),
    "client_email": os.getenv("client_email"),
    "client_id": os.getenv("client_id"),
    "auth_uri": os.getenv("auth_uri"),
    "token_uri": os.getenv("token_uri"),
    "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
    "client_x509_cert_url": os.getenv("client_x509_cert_url"),
}

credentials = service_account.Credentials.from_service_account_info(service_account_info)
db = firestore.Client(credentials=credentials, project=service_account_info["project_id"])

# 2) تحميل نموذج TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
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

# 4) دالة لحساب التشابه باستخدام cosine similarity
def calculate_similarity(img1, img2):
    return cosine_similarity(img1.flatten().reshape(1, -1), img2.flatten().reshape(1, -1))[0][0]

# 5) دالة لتحميل الصور من رابط
def fetch_and_preprocess(url, size=(224,224)):
    try:
        r = requests.get(url)
        r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, size).astype("float32") / 255.0
        return np.expand_dims(img, axis=0)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"خطأ في تحميل الصورة: {e}")

# 6) دالة للاستخراج باستخدام TFLite
def infer(image_np):
    try:
        interpreter.set_tensor(input_details[0]['index'], image_np)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        return float(output.squeeze())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المعالجة: {e}")

# 7) نقطة الدخول الرئيسية للمطابقة
@app.post("/match", response_model=list[MatchOut])
def match_endpoint(payload: list[ItemIn]):
    try:
        docs = db.collection("items").stream()
        found = []
        for doc in docs:
            data = doc.to_dict()
            found.append(ItemIn(id=doc.id, imageUrl=data["imageUrl"]))

        results = []
        for lost in payload:
            lost_img = fetch_and_preprocess(lost.imageUrl)
            for f in found:
                if f.id == lost.id: continue
                f_img = fetch_and_preprocess(f.imageUrl)
                score = calculate_similarity(lost_img, f_img)  # استخدام cosine similarity
                results.append(MatchOut(
                    itemId=lost.id,
                    matchItemId=f.id,
                    confidence=score
                ))

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:5]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة البيانات: {e}")

# 8) حفظ البلاغ
@app.post("/save_report")
def save_report(payload: ItemIn):
    try:
        user_id = "user_id_placeholder"  # استبدل هذا بالـ userId الحقيقي
        imageUrl = payload.imageUrl
        db.collection("reports").add({
            "userId": user_id,
            "itemId": payload.id,
            "imageUrl": imageUrl,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return {"message": "تم حفظ البلاغ بنجاح"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل في حفظ البلاغ: {e}")
