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

def fallback_prediction(mq7, mq3, mq4, mq135):
    if mq3 > 300:
        return {"status":"ABNORMAL","disease":"Possible Diabetes","risk_level":"High","confidence":"65%","advice":"Consult a doctor for blood sugar testing."}
    if mq7 > 350:
        return {"status":"ABNORMAL","disease":"Possible Respiratory Issue","risk_level":"Medium","confidence":"60%","advice":"Avoid smoke and see a doctor."}
    if mq4 > 500:
        return {"status":"ABNORMAL","disease":"Possible Digestive Disorder","risk_level":"Medium","confidence":"58%","advice":"Consider a gastroenterology consultation."}
    if mq135 > 400:
        return {"status":"ABNORMAL","disease":"Possible Liver/Kidney Stress","risk_level":"Medium","confidence":"55%","advice":"Stay hydrated and consult a physician."}
    return {"status":"NORMAL","disease":"No disease detected","risk_level":"Low","confidence":"85%","advice":"Breath profile appears normal."}

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
* { margin:0; padding:0; box-sizing:border-box; }
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: radial-gradient(circle at top, #0d1117, #060810);
    color: #e0e0e0; min-height:100vh; padding:24px;
}
.container { max-width:1500px; margin:auto; }
h1 {
    text-align:center; font-size:2.4rem; margin-bottom:6px;
    background:linear-gradient(45deg,#00d4ff,#00ffaa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.subtitle { text-align:center; color:#555; font-size:0.9rem; margin-bottom:22px; letter-spacing:2px; text-transform:uppercase; }
.status-bar { text-align:center; margin-bottom:28px; font-size:1rem; color:#00ff88; }
.status-bar.error   { color:#ff4444; }
.status-bar.warning { color:#ffaa00; }

.sensor-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:18px; margin-bottom:28px; }
.sensor-card {
    background:linear-gradient(145deg,rgba(30,30,55,0.9),rgba(15,15,35,0.95));
    border-radius:16px; padding:22px; text-align:center;
    border:1px solid rgba(0,212,255,0.2); box-shadow:0 10px 28px rgba(0,0,0,0.5); transition:border-color 0.3s;
}
.sensor-card:hover { border-color:rgba(0,212,255,0.5); }
.sensor-label { font-size:0.9rem; color:#888; text-transform:uppercase; letter-spacing:1px; }
.sensor-value { margin-top:10px; font-size:2.4rem; font-weight:bold; background:linear-gradient(45deg,#00d4ff,#00ffaa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

.ai-panel {
    background:linear-gradient(145deg,rgba(20,20,45,0.97),rgba(10,10,30,0.99));
    border-radius:20px; padding:28px 32px; margin-bottom:28px;
    border:1px solid rgba(0,212,255,0.25); box-shadow:0 0 40px rgba(0,100,255,0.08);
}
.ai-panel-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:22px; flex-wrap:wrap; gap:12px; }
.ai-panel-title { font-size:1.3rem; color:#00d4ff; display:flex; align-items:center; gap:10px; }
.ai-panel-title .dot { width:10px; height:10px; border-radius:50%; background:#00ff88; animation:pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(1.4)} }

#predictBtn {
    background:linear-gradient(135deg,#0066ff,#00d4ff); color:#fff; border:none;
    padding:10px 24px; border-radius:50px; font-size:0.95rem; cursor:pointer;
    font-weight:600; transition:opacity 0.2s,transform 0.1s; letter-spacing:0.5px;
}
#predictBtn:hover  { opacity:0.85; }
#predictBtn:active { transform:scale(0.97); }
#predictBtn:disabled { opacity:0.4; cursor:not-allowed; }

.prediction-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:18px; }
.pred-card { background:rgba(255,255,255,0.03); border-radius:14px; padding:18px; border:1px solid rgba(255,255,255,0.07); text-align:center; }
.pred-card .label { font-size:0.8rem; color:#666; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.pred-card .value { font-size:1.5rem; font-weight:bold; color:#fff; }

.status-NORMAL   { color:#00ff88 !important; }
.status-ABNORMAL { color:#ff4444 !important; }
.risk-Low    { color:#00ff88 !important; }
.risk-Medium { color:#ffaa00 !important; }
.risk-High   { color:#ff4444 !important; }

.advice-box { background:rgba(0,212,255,0.06); border-left:3px solid #00d4ff; border-radius:8px; padding:14px 18px; font-size:0.95rem; color:#ccc; line-height:1.6; }
.advice-box strong { color:#00d4ff; }

.spinner { display:inline-block; width:16px; height:16px; border:2px solid rgba(255,255,255,0.2); border-top-color:#fff; border-radius:50%; animation:spin 0.7s linear infinite; margin-right:8px; vertical-align:middle; }
@keyframes spin { to{transform:rotate(360deg)} }

.history-section { margin-bottom:28px; }
.section-title { font-size:1.1rem; color:#00d4ff; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid rgba(0,212,255,0.15); }
.history-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.history-table th { background:rgba(0,212,255,0.08); color:#00d4ff; padding:10px 14px; text-align:left; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
.history-table td { padding:10px 14px; border-bottom:1px solid rgba(255,255,255,0.04); color:#bbb; }
.history-table tr:hover td { background:rgba(255,255,255,0.02); }

.charts-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:22px; }
.chart-container { aspect-ratio:4/3; background:linear-gradient(145deg,rgba(22,22,42,0.97),rgba(10,10,28,0.99)); border-radius:18px; padding:16px; border:1px solid rgba(0,212,255,0.15); }
.chart-title { text-align:center; margin-bottom:8px; color:#00d4ff; font-size:1rem; }
canvas { width:100%!important; height:100%!important; }
</style>
</head>
<body>
<div class="container">

  <h1>🫁 VOC Disease Detection</h1>
  <p class="subtitle">ESP32 · Real-Time Breath Analysis · Claude AI</p>
  <div class="status-bar" id="status">🔄 Connecting...</div>

  <div class="sensor-grid">
    <div class="sensor-card"><div class="sensor-label">MQ7 — CO</div><div class="sensor-value" id="mq7">—</div></div>
    <div class="sensor-card"><div class="sensor-label">MQ3 — Alcohol</div><div class="sensor-value" id="mq3">—</div></div>
    <div class="sensor-card"><div class="sensor-label">MQ4 — Methane</div><div class="sensor-value" id="mq4">—</div></div>
    <div class="sensor-card"><div class="sensor-label">MQ135 — NH3</div><div class="sensor-value" id="mq135">—</div></div>
    <div class="sensor-card"><div class="sensor-label">VOC Score</div><div class="sensor-value" id="voc">—</div></div>
  </div>

  <div class="ai-panel">
    <div class="ai-panel-header">
      <div class="ai-panel-title"><div class="dot"></div>🤖 AI Disease Prediction</div>
      <button id="predictBtn" onclick="runPrediction()">⚡ Run AI Analysis</button>
    </div>
    <div class="prediction-grid">
      <div class="pred-card"><div class="label">Status</div><div class="value" id="pred-status">—</div></div>
      <div class="pred-card"><div class="label">Disease / Condition</div><div class="value" style="font-size:1.1rem;" id="pred-disease">—</div></div>
      <div class="pred-card"><div class="label">Risk Level</div><div class="value" id="pred-risk">—</div></div>
      <div class="pred-card"><div class="label">Confidence</div><div class="value" id="pred-confidence">—</div></div>
    </div>
    <div class="advice-box">
      <strong>💡 Advice: </strong>
      <span id="pred-advice">Click "Run AI Analysis" to get a health assessment from Claude AI.</span>
    </div>
  </div>

  <div class="history-section">
    <div class="section-title">📋 Recent Predictions</div>
    <table class="history-table">
      <thead><tr><th>Time</th><th>Status</th><th>Disease</th><th>Confidence</th><th>Advice</th></tr></thead>
      <tbody id="historyBody"><tr><td colspan="5" style="color:#555;text-align:center;padding:20px;">No predictions yet</td></tr></tbody>
    </table>
  </div>

  <div class="charts-grid">
    <div class="chart-container"><div class="chart-title">MQ7 — Carbon Monoxide</div><canvas id="mq7Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">MQ3 — Alcohol</div><canvas id="mq3Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">MQ4 — Methane</div><canvas id="mq4Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">MQ135 — NH3</div><canvas id="mq135Chart"></canvas></div>
    <div class="chart-container"><div class="chart-title">VOC Score</div><canvas id="vocChart"></canvas></div>
  </div>

</div>
<script>
// ✅ After deploying to Railway, replace this with your Railway public URL
// Example: const SERVER_URL = "https://voc-project.up.railway.app";
const SERVER_URL = "";
const DEVICE_ID  = "device123";
const MAX_POINTS = 50;
const ids = ["mq7","mq3","mq4","mq135","voc"];
const charts = {};
const history = {mq7:[],mq3:[],mq4:[],mq135:[],voc:[],labels:[]};

function makeChart(id) {
    const ctx = document.getElementById(id+"Chart").getContext("2d");
    return new Chart(ctx,{type:"line",data:{labels:[],datasets:[{data:[],borderColor:"#00d4ff",backgroundColor:"rgba(0,212,255,0.12)",borderWidth:2.5,fill:true,tension:0.4,pointRadius:2}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
    scales:{x:{ticks:{color:"#555",maxTicksLimit:6},grid:{color:"rgba(0,212,255,0.07)"}},y:{beginAtZero:true,ticks:{color:"#555"},grid:{color:"rgba(0,212,255,0.07)"}}},
    plugins:{legend:{display:false}}}});
}
ids.forEach(id => charts[id] = makeChart(id));

function setStatus(msg,type="ok"){
    const el=document.getElementById("status");
    el.textContent=msg;
    el.className="status-bar"+(type==="error"?" error":type==="warning"?" warning":"");
}

async function updateDashboard(){
    try{
        const res=await fetch(`${SERVER_URL}/latest/${DEVICE_ID}`,{method:"GET",mode:"cors",cache:"no-cache"});
        if(res.status===404){setStatus("⏳ Waiting for ESP32 data...","warning");return;}
        if(!res.ok){setStatus(`⚠ Backend error (HTTP ${res.status})`,"error");return;}
        const d=await res.json();
        const mq7=Number(d.MQ7??0),mq3=Number(d.MQ3??0),mq4=Number(d.MQ4??0),mq135=Number(d.MQ135??0),voc=Number(d.VOC??0);
        document.getElementById("mq7").textContent=mq7;
        document.getElementById("mq3").textContent=mq3;
        document.getElementById("mq4").textContent=mq4;
        document.getElementById("mq135").textContent=mq135;
        document.getElementById("voc").textContent=voc.toFixed(2);
        setStatus("✅ LIVE  "+new Date().toLocaleTimeString());
        const label=new Date().toLocaleTimeString();
        history.labels.push(label);
        history.mq7.push(mq7);history.mq3.push(mq3);history.mq4.push(mq4);history.mq135.push(mq135);history.voc.push(voc);
        if(history.labels.length>MAX_POINTS){history.labels.shift();ids.forEach(id=>history[id].shift());}
        ids.forEach(id=>{charts[id].data.labels=history.labels;charts[id].data.datasets[0].data=history[id];charts[id].update("none");});
    }catch(e){setStatus("❌ Cannot reach backend","error");}
}

async function runPrediction(){
    const btn=document.getElementById("predictBtn");
    btn.disabled=true;
    btn.innerHTML='<span class="spinner"></span>Analyzing...';
    document.getElementById("pred-advice").textContent="Claude AI is analyzing your breath data...";
    try{
        const res=await fetch(`${SERVER_URL}/predict/${DEVICE_ID}`,{method:"GET",mode:"cors",cache:"no-cache"});
        if(!res.ok){document.getElementById("pred-advice").textContent=`⚠ Failed (HTTP ${res.status})`;return;}
        const p=await res.json();
        const statusEl=document.getElementById("pred-status");
        statusEl.textContent=p.status; statusEl.className="value status-"+p.status;
        document.getElementById("pred-disease").textContent=p.disease||"—";
        const riskEl=document.getElementById("pred-risk");
        riskEl.textContent=p.risk_level||"—"; riskEl.className="value risk-"+(p.risk_level||"Low");
        document.getElementById("pred-confidence").textContent=p.confidence||"—";
        document.getElementById("pred-advice").textContent=p.advice||"—";
        loadPredictionHistory();
    }catch(e){
        document.getElementById("pred-advice").textContent="❌ Error: "+e.message;
    }finally{
        btn.disabled=false; btn.innerHTML="⚡ Run AI Analysis";
    }
}

async function loadPredictionHistory(){
    try{
        const res=await fetch(`${SERVER_URL}/predictions/${DEVICE_ID}`,{mode:"cors",cache:"no-cache"});
        if(!res.ok)return;
        const preds=await res.json();
        if(!preds.length)return;
        document.getElementById("historyBody").innerHTML=preds.map(p=>`
            <tr>
                <td>${p.timestamp}</td>
                <td class="status-${p.status}">${p.status}</td>
                <td>${p.disease}</td>
                <td>${p.confidence}</td>
                <td style="font-size:0.82rem;">${p.advice}</td>
            </tr>`).join("");
    }catch(e){}
}

updateDashboard();
loadPredictionHistory();
setInterval(updateDashboard,2000);
setInterval(loadPredictionHistory,30000);
</script>
</body>
</html>
"""

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
        mq7=int(data.get("MQ7",0)); mq3=int(data.get("MQ3",0)); mq4=int(data.get("MQ4",0)); mq135=int(data.get("MQ135",0))
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
