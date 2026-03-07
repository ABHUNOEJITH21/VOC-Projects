from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import uuid
import datetime
import requests as req
import json
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---- Database Config ----
MYSQL_URL = os.environ.get("MYSQL_URL", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
MYSQL_HOST = os.environ.get("MYSQLHOST", os.environ.get("MYSQL_HOST", "localhost"))
MYSQL_PORT = os.environ.get("MYSQLPORT", os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQLUSER", os.environ.get("MYSQL_USER", "vocuser"))
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD", os.environ.get("MYSQL_PASSWORD", "1234"))
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", os.environ.get("MYSQLDATABASE", "railway"))

print("DEBUG MYSQL_HOST=" + MYSQL_HOST)
print("DEBUG MYSQL_USER=" + MYSQL_USER)
print("DEBUG MYSQL_DATABASE=" + MYSQL_DATABASE)
print("DEBUG DATABASE_URL=" + DATABASE_URL)
print("DEBUG MYSQL_URL=" + MYSQL_URL)

if DATABASE_URL:
    DB_URI = DATABASE_URL.replace("mysql://", "mysql+pymysql://").replace("mysql+mysqlconnector://", "mysql+pymysql://")
elif MYSQL_URL:
    DB_URI = MYSQL_URL.replace("mysql://", "mysql+pymysql://")
else:
    DB_URI = "mysql+pymysql://" + MYSQL_USER + ":" + MYSQL_PASSWORD + "@" + MYSQL_HOST + ":" + MYSQL_PORT + "/" + MYSQL_DATABASE

print("DEBUG DB_URI=" + DB_URI)

app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---- Models ----
class Device(db.Model):
    __tablename__ = "devices"
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), unique=True, nullable=False)
    device_token = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))

class SensorData(db.Model):
    __tablename__ = "sensor_data"
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=False)
    mq7 = db.Column(db.Integer)
    mq3 = db.Column(db.Integer)
    mq4 = db.Column(db.Integer)
    mq135 = db.Column(db.Integer)
    voc = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.String(100), nullable=False)
    mq7 = db.Column(db.Integer)
    mq3 = db.Column(db.Integer)
    mq4 = db.Column(db.Integer)
    mq135 = db.Column(db.Integer)
    voc = db.Column(db.Float)
    status = db.Column(db.String(20))
    disease = db.Column(db.String(200))
    confidence = db.Column(db.String(20))
    advice = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()
    print("DEBUG tables created ok")

# ---- Helpers ----
def calculate_voc(mq7, mq3, mq4, mq135):
    return round((mq7 + mq3 + mq4 + mq135) / 4, 2)

def fallback_prediction(mq7, mq3, mq4, mq135):
    if mq3 > 300:
        return {"status": "ABNORMAL", "disease": "Possible Diabetes", "risk_level": "High", "confidence": "65%", "advice": "Consult a doctor for blood sugar testing."}
    if mq7 > 350:
        return {"status": "ABNORMAL", "disease": "Possible Respiratory Issue", "risk_level": "Medium", "confidence": "60%", "advice": "Avoid smoke and see a doctor."}
    if mq4 > 500:
        return {"status": "ABNORMAL", "disease": "Possible Digestive Disorder", "risk_level": "Medium", "confidence": "58%", "advice": "Consider a gastroenterology consultation."}
    if mq135 > 400:
        return {"status": "ABNORMAL", "disease": "Possible Liver/Kidney Stress", "risk_level": "Medium", "confidence": "55%", "advice": "Stay hydrated and consult a physician."}
    return {"status": "NORMAL", "disease": "No disease detected", "risk_level": "Low", "confidence": "85%", "advice": "Breath profile appears normal."}

def predict_with_claude(mq7, mq3, mq4, mq135, voc):
    if not ANTHROPIC_API_KEY:
        return fallback_prediction(mq7, mq3, mq4, mq135)
    prompt = "You are a medical AI analyzing breath VOC sensor data. MQ7=" + str(mq7) + " MQ3=" + str(mq3) + " MQ4=" + str(mq4) + " MQ135=" + str(mq135) + " VOC=" + str(voc) + ". Biomarkers: MQ3>300=diabetes, MQ7>350=respiratory, MQ4>500=digestive, MQ135>400=liver/kidney, all<150=normal. Reply ONLY with JSON: {\"status\":\"NORMAL or ABNORMAL\",\"disease\":\"name\",\"risk_level\":\"Low/Medium/High\",\"confidence\":\"72%\",\"advice\":\"one sentence\"}"
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": "claude-sonnet-4-20250514", "max_tokens": 300, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = req.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=15)
        if response.status_code != 200:
            return fallback_prediction(mq7, mq3, mq4, mq135)
        raw = response.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print("Claude error: " + str(e))
        return fallback_prediction(mq7, mq3, mq4, mq135)

# ---- Routes ----
@app.route("/")
def home():
    return jsonify({"status": "running", "message": "VOC Backend with Claude AI"})

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
    if not data:
        return jsonify({"error": "No data"}), 400
    device_id = data.get("device_id")
    token = data.get("token")
    if not device_id or not token:
        return jsonify({"error": "Missing fields"}), 400
    device = Device.query.filter_by(device_id=device_id).first()
    if not device or device.device_token != token:
        return jsonify({"error": "Unauthorized"}), 403
    try:
        mq7 = int(data.get("MQ7", 0))
        mq3 = int(data.get("MQ3", 0))
        mq4 = int(data.get("MQ4", 0))
        mq135 = int(data.get("MQ135", 0))
    except Exception:
        return jsonify({"error": "Invalid values"}), 400
    voc = calculate_voc(mq7, mq3, mq4, mq135)
    db.session.add(SensorData(device_id=device_id, mq7=mq7, mq3=mq3, mq4=mq4, mq135=mq135, voc=voc))
    db.session.commit()
    return jsonify({"status": "success", "voc": voc})

@app.route("/latest/<device_id>", methods=["GET", "OPTIONS"])
def latest(device_id):
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = SensorData.query.filter_by(device_id=device_id).order_by(SensorData.timestamp.desc()).first()
    if not data:
        return jsonify({"error": "No data"}), 404
    return jsonify({"MQ7": data.mq7, "MQ3": data.mq3, "MQ4": data.mq4, "MQ135": data.mq135, "VOC": data.voc, "Timestamp": data.timestamp.strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/predict/<device_id>", methods=["GET", "OPTIONS"])
def predict(device_id):
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data = SensorData.query.filter_by(device_id=device_id).order_by(SensorData.timestamp.desc()).first()
    if not data:
        return jsonify({"error": "No data"}), 404
    result = predict_with_claude(data.mq7, data.mq3, data.mq4, data.mq135, data.voc)
    db.session.add(Prediction(device_id=device_id, mq7=data.mq7, mq3=data.mq3, mq4=data.mq4, mq135=data.mq135, voc=data.voc,
        status=result.get("status", "UNKNOWN"), disease=result.get("disease", "Unknown"),
        confidence=result.get("confidence", "N/A"), advice=result.get("advice", "Consult a doctor.")))
    db.session.commit()
    return jsonify({**result, "sensor": {"MQ7": data.mq7, "MQ3": data.mq3, "MQ4": data.mq4, "MQ135": data.mq135, "VOC": data.voc}, "timestamp": data.timestamp.strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/predictions/<device_id>", methods=["GET"])
def prediction_history(device_id):
    preds = Prediction.query.filter_by(device_id=device_id).order_by(Prediction.timestamp.desc()).limit(10).all()
    return jsonify([{"status": p.status, "disease": p.disease, "confidence": p.confidence, "advice": p.advice, "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for p in preds])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
