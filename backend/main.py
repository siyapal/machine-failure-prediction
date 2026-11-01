"""FastAPI backend: loads the trained model, stores machines + prediction history in Postgres (Neon).
Needs the environment variable DATABASE_URL (Neon connection string).

Local run (from the project root):  uvicorn backend.main:app --reload
Vercel:  api/index.py imports `app` from here.

All routes work both as /predict (local Streamlit dashboard) and /api/predict (Vercel).
"""
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import psycopg2
from fastapi import APIRouter, FastAPI, HTTPException
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

# Optional: lets you keep DATABASE_URL in a local .env file (not needed on Vercel).
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add your Neon connection string as an environment variable.")

MODELS_DIR = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "models")

model = joblib.load(MODELS_DIR / "model.pkl")
with open(MODELS_DIR / "config.json") as f:
    config = json.load(f)
THRESHOLD = config["threshold"]

# Optional: a second model that predicts remaining useful life (RUL) in days.
# If Models/rul_model.pkl does not exist, RUL is simply returned as null.
RUL_PATH = MODELS_DIR / "rul_model.pkl"
rul_model = joblib.load(RUL_PATH) if RUL_PATH.exists() else None


class Features(BaseModel):
    Machine_Type: str
    Operational_Hours: int
    Temperature_C: float
    Vibration_mms: float
    Sound_dB: float
    Oil_Level_pct: float
    Coolant_Level_pct: float
    Power_Consumption_kW: float
    Last_Maintenance_Days_Ago: int
    Maintenance_History_Count: int
    Failure_History_Count: int
    AI_Supervision: int
    Error_Codes_Last_30_Days: int
    Machine_Age: int


class PredictRequest(BaseModel):
    machine_code: str
    features: Features


class AuthIn(BaseModel):
    username: str
    password: str


class AccessIn(BaseModel):
    username: str
    machine_code: str


# ---------------------------------------------------------------- database
@contextmanager
def db():
    """Yields a cursor (rows come back as dicts). Commits on success, rolls back on error, always closes."""
    con = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=10)
    try:
        cur = con.cursor()
        yield cur
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


UPSERT_MACHINE = """INSERT INTO machines (machine_code, data) VALUES (%s, %s)
                    ON CONFLICT (machine_code) DO UPDATE SET data = EXCLUDED.data"""

INSERT_PREDICTION = """INSERT INTO predictions (machine_code, predicted_rul_days, failure_probability,
                       failure_within_7_days, risk_level, created_at) VALUES (%s,%s,%s,%s,%s,%s)"""


def _hash(pw, salt=None):
    salt = salt or os.urandom(8).hex()
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()
    return f"{salt}${h}"


def init_db():
    try:
        with db() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL)")
            cur.execute("""CREATE TABLE IF NOT EXISTS access_log (
                log_id SERIAL PRIMARY KEY, username TEXT NOT NULL, machine_code TEXT NOT NULL,
                checked_at TEXT NOT NULL)""")
            cur.execute("CREATE TABLE IF NOT EXISTS machines (machine_code TEXT PRIMARY KEY, data TEXT NOT NULL)")
            cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
                prediction_id SERIAL PRIMARY KEY, machine_code TEXT NOT NULL,
                predicted_rul_days DOUBLE PRECISION, failure_probability DOUBLE PRECISION,
                failure_within_7_days INTEGER, risk_level TEXT, created_at TEXT NOT NULL)""")
            cur.execute("SELECT COUNT(*) AS n FROM machines")
            if cur.fetchone()["n"] == 0:
                demo = dict(Machine_Type=config["machine_types"][0], Operational_Hours=50000,
                            Temperature_C=50.0, Vibration_mms=8.0, Sound_dB=70.0, Oil_Level_pct=70.0,
                            Coolant_Level_pct=70.0, Power_Consumption_kW=50.0,
                            Last_Maintenance_Days_Ago=180, Maintenance_History_Count=5,
                            Failure_History_Count=1, AI_Supervision=1,
                            Error_Codes_Last_30_Days=2, Machine_Age=5)
                cur.execute("INSERT INTO machines (machine_code, data) VALUES (%s, %s) "
                            "ON CONFLICT (machine_code) DO NOTHING", ("MC_TEST_001", json.dumps(demo)))
    except Exception as e:  # e.g. two cold starts creating tables at the same moment
        print("init_db warning:", e)


def ensure_demo_user():
    """Public demo account so visitors can always log in."""
    try:
        with db() as cur:
            cur.execute("SELECT 1 FROM users WHERE username = 'demo'")
            if not cur.fetchone():
                cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s) "
                            "ON CONFLICT (username) DO NOTHING", ("demo", _hash("demo123")))
    except Exception as e:
        print("ensure_demo_user warning:", e)


init_db()
ensure_demo_user()


# ---------------------------------------------------------------- model
def run_model(features: dict) -> dict:
    row = pd.DataFrame([features])[config["features"]]
    prob = float(model.predict_proba(row)[0, 1])
    rul = float(rul_model.predict(row)[0]) if rul_model is not None else None
    level = "HIGH" if prob >= 0.5 else "MEDIUM" if prob >= THRESHOLD else "LOW"
    return {"failure_probability": round(prob, 4),
            "failure_within_7_days": prob >= THRESHOLD,
            "risk_level": level,
            "predicted_rul_days": None if rul is None else round(rul, 3)}


def seed_demo_data(n=300):
    """Load n real machines from the dataset and store one prediction each.
    NOT called automatically (Vercel does not have the Data folder).
    Run it once from your laptop, with DATABASE_URL pointing to Neon:
        python -c "from backend.main import seed_demo_data; seed_demo_data(300)"
    """
    with db() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM predictions")
        if cur.fetchone()["n"] > 0:
            print("Predictions table already has data, seeding skipped")
            return
    try:
        data_dir = next(p for p in ROOT.iterdir() if p.is_dir() and p.name.lower() == "data")
        raw = next(data_dir.rglob("factory_sensor_data.csv.csv"))
        cleaned = next(data_dir.rglob("cleaned_data.csv"))
        ids = pd.read_csv(raw, usecols=["Machine_ID"])["Machine_ID"]
        clean = pd.read_csv(cleaned, usecols=config["features"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with db() as cur:
            for i in clean.sample(n, random_state=42).index:
                feats = json.loads(clean.loc[[i], config["features"]].to_json(orient="records"))[0]
                res = run_model(feats)
                code = str(ids[i])
                cur.execute(UPSERT_MACHINE, (code, json.dumps(feats)))
                cur.execute(INSERT_PREDICTION,
                            (code, res["predicted_rul_days"], res["failure_probability"],
                             int(res["failure_within_7_days"]), res["risk_level"], now))
        print(f"Seeded {n} demo machines")
    except Exception as e:
        print("Seeding skipped:", e)


# ---------------------------------------------------------------- routes
router = APIRouter()


@router.get("/")
def root():
    return {"status": "ok"}


@router.get("/config")
def get_config():
    return config


@router.get("/machines")
def list_machines():
    with db() as cur:
        cur.execute("SELECT machine_code FROM machines ORDER BY machine_code")
        return [r["machine_code"] for r in cur.fetchall()]


@router.get("/machines/{code}")
def get_machine(code: str):
    with db() as cur:
        cur.execute("SELECT data FROM machines WHERE machine_code = %s", (code,))
        r = cur.fetchone()
    if r is None:
        raise HTTPException(404, "Machine not found")
    return json.loads(r["data"])


@router.post("/predict")
def predict(req: PredictRequest):
    feats = req.features.model_dump()
    if feats["Machine_Type"] not in config["machine_types"]:
        raise HTTPException(422, "Unknown machine type")
    result = run_model(feats)
    with db() as cur:
        cur.execute(UPSERT_MACHINE, (req.machine_code, json.dumps(feats)))
        cur.execute(INSERT_PREDICTION,
                    (req.machine_code, result["predicted_rul_days"], result["failure_probability"],
                     int(result["failure_within_7_days"]), result["risk_level"],
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return result


@router.get("/fleet")
def fleet():
    with db() as cur:
        cur.execute("""SELECT p.machine_code, p.failure_probability, p.risk_level,
                p.predicted_rul_days, p.created_at, m.data
            FROM predictions p JOIN machines m ON m.machine_code = p.machine_code
            WHERE p.prediction_id IN (SELECT MAX(prediction_id) FROM predictions GROUP BY machine_code)""")
        rows = cur.fetchall()
    return [{"machine_code": r["machine_code"],
             "machine_type": json.loads(r["data"])["Machine_Type"],
             "failure_probability": r["failure_probability"], "risk_level": r["risk_level"],
             "predicted_rul_days": r["predicted_rul_days"], "created_at": r["created_at"]}
            for r in rows]


@router.get("/history/{code}")
def history(code: str):
    with db() as cur:
        cur.execute("SELECT * FROM predictions WHERE machine_code = %s ORDER BY prediction_id DESC", (code,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


@router.post("/register")
def register(a: AuthIn):
    username = a.username.strip()
    if not username or not a.password:
        raise HTTPException(422, "Username and password required")
    with db() as cur:
        cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            raise HTTPException(400, "Username already exists")
        cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, _hash(a.password)))
    return {"ok": True}


@router.post("/login")
def login(a: AuthIn):
    username = a.username.strip()
    with db() as cur:
        cur.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
    if not row or _hash(a.password, row["password_hash"].split("$")[0]) != row["password_hash"]:
        raise HTTPException(401, "Wrong username or password")
    return {"ok": True}


@router.post("/access_log")
def add_access(a: AccessIn):
    with db() as cur:
        cur.execute("INSERT INTO access_log (username, machine_code, checked_at) VALUES (%s,%s,%s)",
                    (a.username, a.machine_code, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return {"ok": True}


@router.get("/access_log/{code}")
def get_access(code: str):
    with db() as cur:
        cur.execute("SELECT username, machine_code, checked_at FROM access_log "
                    "WHERE machine_code = %s ORDER BY log_id DESC", (code,))
        rows = cur.fetchall()
    return [dict(r) for r in rows]


app = FastAPI(title="Predictive Maintenance API")
app.include_router(router, include_in_schema=False)  # /predict ... (local Streamlit dashboard)
app.include_router(router, prefix="/api")            # /api/predict ... (Vercel + new frontend)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)