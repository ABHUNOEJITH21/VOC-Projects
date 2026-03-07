from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import uuid
import datetime
import requests as req
import json
import os

# ------------------ Flask Setup ------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ✅ Railway provides DATABASE_URL automatically
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("mysql://"):
        DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    # Local fallback
    DB_USER     = os.environ.get("DB_USER",     "vocuser")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "1234")
    DB_HOST     = os.environ.get("DB_HOST",     "localhost")
    DB_NAME     = os.environ.get("DB_NAME",     "iot_voc")
    app.config['SQLALCHEMY_DATABASE_URI'] = \
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ✅ Set ANTHROPIC_API_KEY as environment variable on Railway
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-XXXXXXXXXXXXXXXXXXXXXXXX")

# ------------------ Database Models ------------------
class Device(db.Model):
    __tablename__ = "devices"
    id           = db.Column(db.Integer, primary_key=True)
    device_id    = db.Column(db.String(100), unique=True, nullable=False)
    device_token = db.Column(db.String(200), nullable=False)
    name         = db.Column(db.String(100))

class SensorData(db.Model):
    __tablename__ = "sensor_data"
    id        = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=False)
    mq7       = db.Column(db.Integer)
    mq3       = db.Column(db.Integer)
    mq4       = db.Column(db.Integer)
    mq135     = db.Column(db.Integer)
    voc       = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Prediction(db.Model):
    __tablename__ = "predictions"
    id         = db.Column(db.Integer, primary_key=True)
    device_id  = db.Column(db.String(100), nullable=False)
    mq7        = db.Column(db.Integer)
    mq3        = db.Column(db.Integer)
    mq4        = db.Column(db.Integer)
    mq135      = db.Column(db.Integer)
    voc        = db.Column(db.Float)
    status     = db.Column(db.String(20))
    disease    = db.Column(db.String(200))
    confidence = db.Column(db.String(20))
    advice     = db.Column(db.Text)
    timestamp  = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()

# ------------------ VOC Calculation ------------------
def calculate_voc(mq7, mq3, mq4, mq135):
    return round((mq7 + mq3 + mq4 + mq135) / 4, 2)

# ------------------ Claude AI Prediction ------------------
def predict_with_claude(mq7, mq3, mq4, mq135, voc):
    prompt = f"""You are a medical AI assistant analyzing human breath VOC sensor data for disease screening.

Sensor Readings:
- MQ7  (Carbon Monoxide): {mq7}  (range 0-4095)
- MQ3  (Alcohol/Ethanol): {mq3}  (range 0-4095)
- MQ4  (Methane): {mq4}  (range 0-4095)
- MQ135 (General VOC): {mq135}  (range 0-4095)
- VOC Score: {voc}

Known biomarkers:
- MQ3 > 300: possible diabetes/ketosis
- MQ7 > 350: possible respiratory issue
- MQ4 > 500: possible digestive disorder
- MQ135 > 400: possible liver/kidney stress
- All < 150: likely normal

Respond ONLY with this exact JSON, no extra text:
{{
  "status": "NORMAL" or "ABNORMAL",
  "disease": "short name or No disease detected",
  "risk_level": "Low" or "Medium" or "High",
  "confidence": "like 72%",
  "advice": "one short actionable sentence"
}}"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = req.post("https://api.anthropic.com/v1/messages",
                            headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            return fallback_prediction(mq7, mq3, mq4, mq135)

        raw = response.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())

    except Exception as e:
        print(f"❌ Claude error: {e}")
        return fallback_prediction(mq7, mq3, mq4, mq135)


def fallback_prediction(mq7, mq3, mq4, mq135):
    if mq3 > 300:
        return {"status":"ABNORMAL","disease":"Possible Diabetes (Acetone)","risk_level":"High","confidence":"65%","advice":"Consult a doctor for blood sugar testing."}
    elif mq7 > 350:
        return {"status":"ABNORMAL","disease":"Possible Respiratory Issue","risk_level":"Medium","confidence":"60%","advice":"Avoid smoke exposure and see a doctor."}
    elif mq4 > 500:
        return {"status":"ABNORMAL","disease":"Possible Digestive Disorder","risk_level":"Medium","confidence":"58%","advice":"Consider a gastroenterology consultation."}
    elif mq135 > 400:
        return {"status":"ABNORMAL","disease":"Possible Liver/Kidney Stress","risk_level":"Medium","confidence":"55%","advice":"Stay hydrated and consult a physician."}
    else:
        return {"status":"NORMAL","disease":"No disease detected","risk_level":"Low","confidence":"85%","advice":"Breath profile appears normal. Maintain healthy habits."}


# ------------------ Routes ------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    if not data or "device_id" not in data:
        return jsonify({"error": "device_id required"}), 400
    existing = Device.query.filter_by(device_id=data["device_id"]).first()
    if existing:
        return jsonify({"token": existing.device_token})
    token = str(uuid.uuid4())
    db.session.add(Device(device_id=data["device_id"], device_token=token, name="VOC Device"))
    db.session.commit()
    return jsonify({"token": token})


@app.route("/data", methods=["POST"])
def receive_data():
    data = request.json
    if not data: return jsonify({"error": "No data"}), 400
    device_id, token = data.get("device_id"), data.get("token")
    if not device_id or not token: return jsonify({"error": "Missing fields"}), 400
    device = Device.query.filter_by(device_id=device_id).first()
    if not device or device.device_token != token: return jsonify({"error": "Unauthorized"}), 403
    try:
        mq7,mq3,mq4,mq135 = int(data.get("MQ7",0)),int(data.get("MQ3",0)),int(data.get("MQ4",0)),int(data.get("MQ135",0))
    except:
        return jsonify({"error": "Invalid values"}), 400
    voc = calculate_voc(mq7,mq3,mq4,mq135)
    db.session.add(SensorData(device_id=device_id,mq7=mq7,mq3=mq3,mq4=mq4,mq135=mq135,voc=voc))
    db.session.commit()
    return jsonify({"status": "success", "voc": voc})


@app.route("/latest/<device_id>", methods=["GET","OPTIONS"])
def latest(device_id):
    if request.method == "OPTIONS": return jsonify({}), 200
    data = SensorData.query.filter_by(device_id=device_id).order_by(SensorData.timestamp.desc()).first()
    if not data: return jsonify({"error": "No data"}), 404
    return jsonify({"MQ7":data.mq7,"MQ3":data.mq3,"MQ4":data.mq4,"MQ135":data.mq135,"VOC":data.voc,"Timestamp":data.timestamp.strftime("%Y-%m-%d %H:%M:%S")})


@app.route("/predict/<device_id>", methods=["GET","OPTIONS"])
def predict(device_id):
    if request.method == "OPTIONS": return jsonify({}), 200
    data = SensorData.query.filter_by(device_id=device_id).order_by(SensorData.timestamp.desc()).first()
    if not data: return jsonify({"error": "No data"}), 404
    result = predict_with_claude(data.mq7,data.mq3,data.mq4,data.mq135,data.voc)
    db.session.add(Prediction(device_id=device_id,mq7=data.mq7,mq3=data.mq3,mq4=data.mq4,mq135=data.mq135,voc=data.voc,
        status=result.get("status","UNKNOWN"),disease=result.get("disease","Unknown"),
        confidence=result.get("confidence","N/A"),advice=result.get("advice","Consult a doctor.")))
    db.session.commit()
    return jsonify({**result, "sensor":{"MQ7":data.mq7,"MQ3":data.mq3,"MQ4":data.mq4,"MQ135":data.mq135,"VOC":data.voc},
                    "timestamp":data.timestamp.strftime("%Y-%m-%d %H:%M:%S")})


@app.route("/predictions/<device_id>", methods=["GET"])
def prediction_history(device_id):
    preds = Prediction.query.filter_by(device_id=device_id).order_by(Prediction.timestamp.desc()).limit(10).all()
    return jsonify([{"status":p.status,"disease":p.disease,"confidence":p.confidence,"advice":p.advice,"timestamp":p.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for p in preds])


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status":"running","message":"🚀 IoT VOC Backend with Claude AI"})


# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
