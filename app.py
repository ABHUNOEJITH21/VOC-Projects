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
RAW_URL = (
    os.environ.get("MYSQL_URL") or
    os.environ.get("DATABASE_URL") or
    ""
)

if RAW_URL:
    DB_URI = RAW_URL
    DB_URI = DB_URI.replace("mysql+mysqlconnector://", "mysql+pymysql://")
    if DB_URI.startswith("mysql://"):
        DB_URI = "mysql+pymysql://" + DB_URI[len("mysql://"):]
else:
    H = os.environ.get("MYSQLHOST", "localhost")
    P = os.environ.get("MYSQLPORT", "3306")
    U = os.environ.get("MYSQLUSER", "vocuser")
    W = os.environ.get("MYSQLPASSWORD", "1234")
    D = os.environ.get("MYSQL_DATABASE", "railway")
    DB_URI = "mysql+pymysql://" + U + ":" + W + "@" + H + ":" + P + "/" + D

app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 300}

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
    print("DEBUG tables ok")

# ---- Helpers ----
def calculate_voc(mq7, mq3, mq4, mq135):
    return round((mq7 + mq3 + mq4 + mq135) / 4, 2)

# ---- Advanced Rule-Based Fallback ----
def fallback_prediction(mq7, mq3, mq4, mq135):
    diseases = []

    # Score each condition based on sensor levels
    # MQ3 - Alcohol/Acetone → Diabetes, Ketosis, Alcohol
    if mq3 > 600:
        diseases.append({"disease": "Severe Diabetes / Diabetic Ketoacidosis", "confidence": 88, "risk_level": "High", "sensor": "MQ3=" + str(mq3)})
    elif mq3 > 400:
        diseases.append({"disease": "Possible Diabetes / Ketosis", "confidence": 75, "risk_level": "High", "sensor": "MQ3=" + str(mq3)})
    elif mq3 > 300:
        diseases.append({"disease": "Early Diabetes Indicator (Acetone)", "confidence": 62, "risk_level": "Medium", "sensor": "MQ3=" + str(mq3)})
    elif mq3 > 200:
        diseases.append({"disease": "Mild Ketosis / Diet-Induced", "confidence": 50, "risk_level": "Low", "sensor": "MQ3=" + str(mq3)})

    # MQ7 - Carbon Monoxide → Respiratory, Lung Issues
    if mq7 > 700:
        diseases.append({"disease": "Severe Respiratory / Lung Disease", "confidence": 87, "risk_level": "High", "sensor": "MQ7=" + str(mq7)})
    elif mq7 > 500:
        diseases.append({"disease": "Chronic Obstructive Pulmonary Disease (COPD)", "confidence": 74, "risk_level": "High", "sensor": "MQ7=" + str(mq7)})
    elif mq7 > 350:
        diseases.append({"disease": "Respiratory Infection / Lung Inflammation", "confidence": 63, "risk_level": "Medium", "sensor": "MQ7=" + str(mq7)})
    elif mq7 > 250:
        diseases.append({"disease": "Mild Respiratory Stress", "confidence": 48, "risk_level": "Low", "sensor": "MQ7=" + str(mq7)})

    # MQ4 - Methane → Digestive, Gut Disorders
    if mq4 > 800:
        diseases.append({"disease": "Severe Intestinal Obstruction / IBS", "confidence": 85, "risk_level": "High", "sensor": "MQ4=" + str(mq4)})
    elif mq4 > 600:
        diseases.append({"disease": "Irritable Bowel Syndrome (IBS)", "confidence": 72, "risk_level": "Medium", "sensor": "MQ4=" + str(mq4)})
    elif mq4 > 500:
        diseases.append({"disease": "Digestive Disorder / Gut Imbalance", "confidence": 61, "risk_level": "Medium", "sensor": "MQ4=" + str(mq4)})
    elif mq4 > 350:
        diseases.append({"disease": "Mild Digestive Issue", "confidence": 45, "risk_level": "Low", "sensor": "MQ4=" + str(mq4)})

    # MQ135 - General VOC / Ammonia → Liver, Kidney
    if mq135 > 700:
        diseases.append({"disease": "Severe Liver / Kidney Failure", "confidence": 86, "risk_level": "High", "sensor": "MQ135=" + str(mq135)})
    elif mq135 > 550:
        diseases.append({"disease": "Chronic Kidney Disease (CKD)", "confidence": 73, "risk_level": "High", "sensor": "MQ135=" + str(mq135)})
    elif mq135 > 400:
        diseases.append({"disease": "Liver / Kidney Metabolic Stress", "confidence": 62, "risk_level": "Medium", "sensor": "MQ135=" + str(mq135)})
    elif mq135 > 280:
        diseases.append({"disease": "Mild Liver Stress / Dehydration", "confidence": 47, "risk_level": "Low", "sensor": "MQ135=" + str(mq135)})

    # Combined patterns — multi-sensor correlations
    if mq3 > 300 and mq135 > 400:
        diseases.append({"disease": "Diabetic Nephropathy (Diabetes + Kidney)", "confidence": 82, "risk_level": "High", "sensor": "MQ3+MQ135"})
    if mq7 > 350 and mq135 > 400:
        diseases.append({"disease": "Lung-Liver Complication", "confidence": 78, "risk_level": "High", "sensor": "MQ7+MQ135"})
    if mq4 > 400 and mq135 > 350:
        diseases.append({"disease": "Gut-Liver Axis Disorder", "confidence": 70, "risk_level": "Medium", "sensor": "MQ4+MQ135"})
    if mq7 > 300 and mq4 > 400:
        diseases.append({"disease": "Pulmonary + Digestive Complication", "confidence": 68, "risk_level": "Medium", "sensor": "MQ7+MQ4"})

    if not diseases:
        return {
            "status": "NORMAL",
            "disease": "No disease detected",
            "risk_level": "Low",
            "confidence": "92%",
            "advice": "Breath profile is within normal range. Stay hydrated and maintain healthy habits.",
            "all_predictions": []
        }

    # Sort by confidence — highest first
    diseases.sort(key=lambda x: x["confidence"], reverse=True)
    top = diseases[0]

    return {
        "status": "ABNORMAL",
        "disease": top["disease"],
        "risk_level": top["risk_level"],
        "confidence": str(top["confidence"]) + "%",
        "advice": get_advice(top["disease"], top["risk_level"]),
        "all_predictions": [
            {"disease": d["disease"], "confidence": str(d["confidence"]) + "%", "risk_level": d["risk_level"]}
            for d in diseases[:5]
        ]
    }

def get_advice(disease, risk):
    advice_map = {
        "Diabetes": "Monitor blood glucose levels immediately and consult an endocrinologist.",
        "Ketosis": "Check blood sugar levels and increase carbohydrate intake if non-diabetic.",
        "COPD": "Seek immediate pulmonology consultation and avoid all smoke exposure.",
        "Respiratory": "Rest, avoid polluted air, and consult a pulmonologist if persistent.",
        "Kidney": "Increase water intake, reduce protein, and consult a nephrologist urgently.",
        "Liver": "Avoid alcohol, fatty foods, and consult a gastroenterologist.",
        "IBS": "Follow a low-FODMAP diet and consult a gastroenterologist.",
        "Digestive": "Stay hydrated, eat fiber-rich foods, and consult a doctor.",
        "Gut": "Maintain probiotic diet and consult a gastroenterologist.",
        "Lung": "Seek immediate medical attention for breathing difficulties.",
        "Nephropathy": "Urgent consultation with both endocrinologist and nephrologist required.",
    }
    for keyword, advice in advice_map.items():
        if keyword.lower() in disease.lower():
            return advice
    if risk == "High":
        return "Elevated VOC levels detected. Seek medical consultation immediately."
    elif risk == "Medium":
        return "Abnormal breath markers detected. Schedule a doctor visit soon."
    return "Mild abnormality detected. Monitor your health and stay hydrated."

# ---- Claude AI Prediction ----
def predict_with_claude(mq7, mq3, mq4, mq135, voc):
    if not ANTHROPIC_API_KEY:
        return fallback_prediction(mq7, mq3, mq4, mq135)

    prompt = """You are an expert medical AI specializing in breath biomarker analysis and VOC-based disease detection.

Sensor readings from human breath analysis (ADC values 0-4095, 12-bit resolution):
- MQ7  (Carbon Monoxide sensor): """ + str(mq7) + """
- MQ3  (Alcohol/Acetone/Ethanol sensor): """ + str(mq3) + """
- MQ4  (Methane/Natural Gas sensor): """ + str(mq4) + """
- MQ135 (Ammonia/VOC/Air Quality sensor): """ + str(mq135) + """
- Calculated VOC Score (average): """ + str(voc) + """

Detailed biomarker reference thresholds (ADC values):
MQ3 (Acetone/Ethanol):
  - 150-200: Baseline normal breath
  - 200-300: Mild elevated — possible diet-induced ketosis
  - 300-400: Early diabetes indicator / alcohol consumption
  - 400-600: Probable diabetes / diabetic ketosis
  - 600+: Severe diabetes / diabetic ketoacidosis

MQ7 (Carbon Monoxide):
  - 100-200: Normal baseline
  - 200-300: Mild respiratory stress / smoker
  - 300-500: Respiratory infection / early COPD
  - 500-700: Chronic Obstructive Pulmonary Disease (COPD)
  - 700+: Severe lung disease / carbon monoxide poisoning

MQ4 (Methane):
  - 200-350: Normal gut flora
  - 350-500: Mild digestive imbalance
  - 500-650: Irritable Bowel Syndrome / gut disorder
  - 650-800: Severe IBS / intestinal obstruction
  - 800+: Critical digestive pathology

MQ135 (Ammonia/VOC):
  - 150-280: Normal metabolic baseline
  - 280-400: Mild liver/kidney stress / dehydration
  - 400-550: Liver dysfunction / early kidney disease
  - 550-700: Chronic Kidney Disease (CKD) / liver disease
  - 700+: Severe kidney/liver failure / uremia

Combined pattern analysis:
- High MQ3 + High MQ135: Diabetic Nephropathy
- High MQ7 + High MQ135: Pulmonary-Hepatic syndrome
- High MQ4 + High MQ135: Gut-Liver axis disorder
- High MQ7 + High MQ4: Pulmonary-Digestive complication
- All sensors high: Systemic metabolic disorder
- All sensors low (<200): Healthy normal breath

Analyze ALL sensor values carefully and provide:
1. The PRIMARY disease prediction (highest confidence)
2. Up to 4 ADDITIONAL possible conditions ranked by confidence
3. Confidence must reflect actual sensor deviation from normal (higher deviation = higher confidence)
4. Risk level: Low (<40% confidence), Medium (40-70%), High (>70%)

Respond ONLY with this exact JSON format, no extra text:
{
  "status": "NORMAL or ABNORMAL",
  "disease": "primary disease name",
  "risk_level": "Low or Medium or High",
  "confidence": "percentage like 87%",
  "advice": "specific actionable medical advice in one sentence",
  "all_predictions": [
    {"disease": "disease name", "confidence": "87%", "risk_level": "High"},
    {"disease": "disease name", "confidence": "74%", "risk_level": "Medium"},
    {"disease": "disease name", "confidence": "61%", "risk_level": "Medium"},
    {"disease": "disease name", "confidence": "45%", "risk_level": "Low"}
  ]
}"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = req.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=20)
        if response.status_code != 200:
            print("Claude API error: " + str(response.status_code))
            return fallback_prediction(mq7, mq3, mq4, mq135)
        raw = response.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print("Claude error: " + str(e))
        return fallback_prediction(mq7, mq3, mq4, mq135)

# ---- Dashboard HTML ----
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VOC Disease Detection — AI Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:radial-gradient(circle at top,#0d1117,#060810);color:#e0e0e0;min-height:100vh;padding:24px}
.container{max-width:1500px;margin:auto}
h1{text-align:center;font-size:2.4rem;margin-bottom:6px;background:linear-gradient(45deg,#00d4ff,#00ffaa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#555;font-size:.9rem;margin-bottom:22px;letter-spacing:2px;text-transform:uppercase}
.status-bar{text-align:center;margin-bottom:28px;font-size:1rem;color:#00ff88}
.status-bar.error{color:#ff4444}.status-bar.warning{color:#ffaa00}
.sensor-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;margin-bottom:28px}
.sensor-card{background:linear-gradient(145deg,rgba(30,30,55,.9),rgba(15,15,35,.95));border-radius:16px;padding:22px;text-align:center;border:1px solid rgba(0,212,255,.2);box-shadow:0 10px 28px rgba(0,0,0,.5);transition:border-color .3s}
.sensor-card:hover{border-color:rgba(0,212,255,.5)}
.sensor-label{font-size:.9rem;color:#888;text-transform:uppercase;letter-spacing:1px}
.sensor-value{margin-top:10px;font-size:2.4rem;font-weight:bold;background:linear-gradient(45deg,#00d4ff,#00ffaa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.ai-panel{background:linear-gradient(145deg,rgba(20,20,45,.97),rgba(10,10,30,.99));border-radius:20px;padding:28px 32px;margin-bottom:28px;border:1px solid rgba(0,212,255,.25)}
.ai-panel-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;flex-wrap:wrap;gap:12px}
.ai-panel-title{font-size:1.3rem;color:#00d4ff;display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;background:#00ff88;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.4)}}
#predictBtn{background:linear-gradient(135deg,#0066ff,#00d4ff);color:#fff;border:none;padding:10px 24px;border-radius:50px;font-size:.95rem;cursor:pointer;font-weight:600;transition:opacity .2s,transform .1s}
#predictBtn:hover{opacity:.85}#predictBtn:active{transform:scale(.97)}#predictBtn:disabled{opacity:.4;cursor:not-allowed}
.prediction-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:18px}
.pred-card{background:rgba(255,255,255,.03);border-radius:14px;padding:18px;border:1px solid rgba(255,255,255,.07);text-align:center}
.pred-card .label{font-size:.8rem;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.pred-card .value{font-size:1.5rem;font-weight:bold;color:#fff}
.status-NORMAL{color:#00ff88!important}.status-ABNORMAL{color:#ff4444!important}
.risk-Low{color:#00ff88!important}.risk-Medium{color:#ffaa00!important}.risk-High{color:#ff4444!important}
.advice-box{background:rgba(0,212,255,.06);border-left:3px solid #00d4ff;border-radius:8px;padding:14px 18px;font-size:.95rem;color:#ccc;line-height:1.6;margin-bottom:18px}
.advice-box strong{color:#00d4ff}

/* All predictions table */
.all-pred-section{margin-top:16px}
.all-pred-title{font-size:.85rem;color:#00d4ff;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px}
.pred-row{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-radius:10px;margin-bottom:6px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05)}
.pred-row-name{font-size:.9rem;color:#ccc;flex:1}
.pred-row-conf{font-size:.85rem;font-weight:bold;margin:0 12px;min-width:45px;text-align:right}
.pred-row-risk{font-size:.75rem;padding:3px 10px;border-radius:20px;font-weight:600}
.risk-badge-Low{background:rgba(0,255,136,.1);color:#00ff88;border:1px solid rgba(0,255,136,.3)}
.risk-badge-Medium{background:rgba(255,170,0,.1);color:#ffaa00;border:1px solid rgba(255,170,0,.3)}
.risk-badge-High{background:rgba(255,68,68,.1);color:#ff4444;border:1px solid rgba(255,68,68,.3)}

/* Confidence bar */
.conf-bar-wrap{flex:1;margin:0 12px;height:6px;background:rgba(255,255,255,.08);border-radius:3px;max-width:120px}
.conf-bar{height:100%;border-radius:3px;transition:width .5s}
.conf-bar-high{background:linear-gradient(90deg,#ff4444,#ff6666)}
.conf-bar-medium{background:linear-gradient(90deg,#ffaa00,#ffcc44)}
.conf-bar-low{background:linear-gradient(90deg,#00d4ff,#00ffaa)}

.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.2);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;margin-right:8px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}
.history-section{margin-bottom:28px}
.section-title{font-size:1.1rem;color:#00d4ff;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid rgba(0,212,255,.15)}
.history-table{width:100%;border-collapse:collapse;font-size:.88rem}
.history-table th{background:rgba(0,212,255,.08);color:#00d4ff;padding:10px 14px;text-align:left;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.history-table td{padding:10px 14px;border-bottom:1px solid rgba(255,255,255,.04);color:#bbb}
.history-table tr:hover td{background:rgba(255,255,255,.02)}
.charts-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:22px}
.chart-container{aspect-ratio:4/3;background:linear-gradient(145deg,rgba(22,22,42,.97),rgba(10,10,28,.99));border-radius:18px;padding:16px;border:1px solid rgba(0,212,255,.15)}
.chart-title{text-align:center;margin-bottom:8px;color:#00d4ff;font-size:1rem}
canvas{width:100%!important;height:100%!important}
</style>
</head>
<body>
<div class="container">
  <h1>&#129753; VOC Disease Detection</h1>
  <p class="subtitle">ESP32 &middot; Real-Time Breath Analysis &middot; Claude AI</p>
  <div class="status-bar" id="status">&#128260; Connecting...</div>
  <div class="sensor-grid">
    <div class="sensor-card"><div class="sensor-label">MQ7 &mdash; CO</div><div class="sensor-value" id="mq7">&mdash;</div></div>
    <div class="sensor-card"><div class="sensor-label">MQ3 &mdash; Alcohol</div><div class="sensor-value" id="mq3">&mdash;</div></div>
    <div class="sensor-card"><div class="sensor-label">MQ4 &mdash; Methane</div><div class="sensor-value" id="mq4">&mdash;</div></div>
    <div class="sensor-card"><div class="sensor-label">MQ135 &mdash; VOC</div><div class="sensor-value" id="mq135">&mdash;</div></div>
    <div class="sensor-card"><div class="sensor-label">VOC Score</div><div class="sensor-value" id="voc">&mdash;</div></div>
  </div>
  <div class="ai-panel">
    <div class="ai-panel-header">
      <div class="ai-panel-title"><div class="dot"></div>&#129504; Claude AI Disease Prediction</div>
      <button id="predictBtn" onclick="runPrediction()">&#9889; Run AI Analysis</button>
    </div>
    <div class="prediction-grid">
      <div class="pred-card"><div class="label">Status</div><div class="value" id="pred-status">&mdash;</div></div>
      <div class="pred-card"><div class="label">Primary Disease</div><div class="value" style="font-size:1rem" id="pred-disease">&mdash;</div></div>
      <div class="pred-card"><div class="label">Risk Level</div><div class="value" id="pred-risk">&mdash;</div></div>
      <div class="pred-card"><div class="label">Confidence</div><div class="value" id="pred-confidence">&mdash;</div></div>
    </div>
    <div class="advice-box"><strong>&#128161; Advice: </strong><span id="pred-advice">Click "Run AI Analysis" to get a detailed health assessment from Claude AI.</span></div>

    <!-- All Predictions Section -->
    <div class="all-pred-section" id="allPredSection" style="display:none">
      <div class="all-pred-title">&#128202; All Possible Conditions (Ranked by Confidence)</div>
      <div id="allPredList"></div>
    </div>
  </div>

  <div class="history-section">
    <div class="section-title">&#128203; Recent Predictions</div>
    <table class="history-table">
      <thead><tr><th>Time</th><th>Status</th><th>Disease</th><th>Confidence</th><th>Advice</th></tr></thead>
      <tbody id="historyBody"><tr><td colspan="5" style="color:#555;text-align:center;padding:20px">No predictions yet</td></tr></tbody>
    </table>
  </div>

  <div class="charts-grid">
    <div class="chart-container"><div class="chart-title">MQ7 &mdash; Carbon Monoxide</div><canvas id="mq7Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">MQ3 &mdash; Alcohol/Acetone</div><canvas id="mq3Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">MQ4 &mdash; Methane</div><canvas id="mq4Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">MQ135 &mdash; VOC/Ammonia</div><canvas id="mq135Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">VOC Score</div><canvas id="vocChart"></canvas></div>
  </div>
</div>
<script>
// ✅ Empty string = same-origin (served by Flask on Railway)
const SERVER_URL="";
const DEVICE_ID="device123",MAX_POINTS=50,ids=["mq7","mq3","mq4","mq135","voc"],charts={},hist={mq7:[],mq3:[],mq4:[],mq135:[],voc:[],labels:[]};

function makeChart(id){const ctx=document.getElementById(id+"Chart").getContext("2d");return new Chart(ctx,{type:"line",data:{labels:[],datasets:[{data:[],borderColor:"#00d4ff",backgroundColor:"rgba(0,212,255,0.12)",borderWidth:2.5,fill:true,tension:0.4,pointRadius:2}]},options:{responsive:true,maintainAspectRatio:false,animation:false,scales:{x:{ticks:{color:"#555",maxTicksLimit:6},grid:{color:"rgba(0,212,255,0.07)"}},y:{beginAtZero:true,ticks:{color:"#555"},grid:{color:"rgba(0,212,255,0.07)"}}},plugins:{legend:{display:false}}}})}
ids.forEach(id=>charts[id]=makeChart(id));

function setStatus(msg,type="ok"){const el=document.getElementById("status");el.textContent=msg;el.className="status-bar"+(type==="error"?" error":type==="warning"?" warning":"")}

async function updateDashboard(){
    try{
        const res=await fetch("/latest/"+DEVICE_ID,{cache:"no-cache"});
        if(res.status===404){setStatus("Waiting for ESP32 data...","warning");return;}
        if(!res.ok){setStatus("Backend error HTTP "+res.status,"error");return;}
        const d=await res.json();
        const mq7=Number(d.MQ7??0),mq3=Number(d.MQ3??0),mq4=Number(d.MQ4??0),mq135=Number(d.MQ135??0),voc=Number(d.VOC??0);
        document.getElementById("mq7").textContent=mq7;
        document.getElementById("mq3").textContent=mq3;
        document.getElementById("mq4").textContent=mq4;
        document.getElementById("mq135").textContent=mq135;
        document.getElementById("voc").textContent=voc.toFixed(2);
        setStatus("LIVE "+new Date().toLocaleTimeString());
        const label=new Date().toLocaleTimeString();
        hist.labels.push(label);hist.mq7.push(mq7);hist.mq3.push(mq3);hist.mq4.push(mq4);hist.mq135.push(mq135);hist.voc.push(voc);
        if(hist.labels.length>MAX_POINTS){hist.labels.shift();ids.forEach(id=>hist[id].shift())}
        ids.forEach(id=>{charts[id].data.labels=hist.labels;charts[id].data.datasets[0].data=hist[id];charts[id].update("none")});
    }catch(e){setStatus("Cannot reach backend","error")}
}

function getBarClass(risk){return risk==="High"?"conf-bar-high":risk==="Medium"?"conf-bar-medium":"conf-bar-low"}

async function runPrediction(){
    const btn=document.getElementById("predictBtn");
    btn.disabled=true;btn.innerHTML='<span class="spinner"></span>Analyzing...';
    document.getElementById("pred-advice").textContent="Claude AI is analyzing all VOC biomarkers...";
    document.getElementById("allPredSection").style.display="none";
    try{
        const res=await fetch("/predict/"+DEVICE_ID,{cache:"no-cache"});
        if(!res.ok){document.getElementById("pred-advice").textContent="Failed HTTP "+res.status;return;}
        const p=await res.json();

        const se=document.getElementById("pred-status");
        se.textContent=p.status;se.className="value status-"+p.status;
        document.getElementById("pred-disease").textContent=p.disease||"—";
        const re=document.getElementById("pred-risk");
        re.textContent=p.risk_level||"—";re.className="value risk-"+(p.risk_level||"Low");
        document.getElementById("pred-confidence").textContent=p.confidence||"—";
        document.getElementById("pred-advice").textContent=p.advice||"—";

        // Show all predictions
        if(p.all_predictions && p.all_predictions.length > 0){
            const list=document.getElementById("allPredList");
            list.innerHTML=p.all_predictions.map(pred=>{
                const confNum=parseInt(pred.confidence)||0;
                const barClass=getBarClass(pred.risk_level);
                return '<div class="pred-row">'
                    +'<span class="pred-row-name">'+pred.disease+'</span>'
                    +'<div class="conf-bar-wrap"><div class="conf-bar '+barClass+'" style="width:'+confNum+'%"></div></div>'
                    +'<span class="pred-row-conf risk-'+pred.risk_level+'">'+pred.confidence+'</span>'
                    +'<span class="pred-row-risk risk-badge-'+pred.risk_level+'">'+pred.risk_level+'</span>'
                    +'</div>';
            }).join("");
            document.getElementById("allPredSection").style.display="block";
        }

        loadPredictionHistory();
    }catch(e){
        document.getElementById("pred-advice").textContent="Error: "+e.message;
    }finally{btn.disabled=false;btn.innerHTML="&#9889; Run AI Analysis"}
}

async function loadPredictionHistory(){
    try{
        const res=await fetch("/predictions/"+DEVICE_ID,{cache:"no-cache"});
        if(!res.ok)return;
        const preds=await res.json();
        if(!preds.length)return;
        document.getElementById("historyBody").innerHTML=preds.map(p=>
            "<tr><td>"+p.timestamp+"</td><td class='status-"+p.status+"'>"+p.status+"</td><td>"+p.disease+"</td><td>"+p.confidence+"</td><td style='font-size:.82rem'>"+p.advice+"</td></tr>"
        ).join("");
    }catch(e){}
}

updateDashboard();loadPredictionHistory();
setInterval(updateDashboard,2000);setInterval(loadPredictionHistory,30000);
</script>
</body>
</html>"""

# ---- Routes ----
@app.route("/")
def dashboard():
    return DASHBOARD_HTML, 200, {"Content-Type": "text/html"}

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
    device_id = data.get("device_id")
    token = data.get("token")
    if not device_id or not token: return jsonify({"error": "Missing fields"}), 400
    device = Device.query.filter_by(device_id=device_id).first()
    if not device or device.device_token != token: return jsonify({"error": "Unauthorized"}), 403
    try:
        mq7=int(data.get("MQ7",0));mq3=int(data.get("MQ3",0));mq4=int(data.get("MQ4",0));mq135=int(data.get("MQ135",0))
    except Exception:
        return jsonify({"error": "Invalid values"}), 400
    voc = calculate_voc(mq7, mq3, mq4, mq135)
    db.session.add(SensorData(device_id=device_id, mq7=mq7, mq3=mq3, mq4=mq4, mq135=mq135, voc=voc))
    db.session.commit()
    return jsonify({"status": "success", "voc": voc})

@app.route("/latest/<device_id>", methods=["GET", "OPTIONS"])
def latest(device_id):
    if request.method == "OPTIONS": return jsonify({}), 200
    data = SensorData.query.filter_by(device_id=device_id).order_by(SensorData.timestamp.desc()).first()
    if not data: return jsonify({"error": "No data"}), 404
    return jsonify({"MQ7":data.mq7,"MQ3":data.mq3,"MQ4":data.mq4,"MQ135":data.mq135,"VOC":data.voc,"Timestamp":data.timestamp.strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/predict/<device_id>", methods=["GET", "OPTIONS"])
def predict(device_id):
    if request.method == "OPTIONS": return jsonify({}), 200
    data = SensorData.query.filter_by(device_id=device_id).order_by(SensorData.timestamp.desc()).first()
    if not data: return jsonify({"error": "No data"}), 404
    result = predict_with_claude(data.mq7, data.mq3, data.mq4, data.mq135, data.voc)
    db.session.add(Prediction(device_id=device_id, mq7=data.mq7, mq3=data.mq3, mq4=data.mq4, mq135=data.mq135, voc=data.voc,
        status=result.get("status","UNKNOWN"), disease=result.get("disease","Unknown"),
        confidence=result.get("confidence","N/A"), advice=result.get("advice","Consult a doctor.")))
    db.session.commit()
    return jsonify({**result,"sensor":{"MQ7":data.mq7,"MQ3":data.mq3,"MQ4":data.mq4,"MQ135":data.mq135,"VOC":data.voc},"timestamp":data.timestamp.strftime("%Y-%m-%d %H:%M:%S")})

@app.route("/predictions/<device_id>", methods=["GET"])
def prediction_history(device_id):
    preds = Prediction.query.filter_by(device_id=device_id).order_by(Prediction.timestamp.desc()).limit(10).all()
    return jsonify([{"status":p.status,"disease":p.disease,"confidence":p.confidence,"advice":p.advice,"timestamp":p.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for p in preds])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
