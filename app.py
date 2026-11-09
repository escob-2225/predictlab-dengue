import os, io, json, sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "dengue.db")
MODEL_PATH = os.path.join(BASE, "models", "modelos_entrenados.joblib")
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "predictlab-dengue-local")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

bundle = joblib.load(MODEL_PATH)
MODELS = bundle["models"]
FEATURES = bundle["features"]
METRICS = bundle["metrics"]
MATRICES = bundle["matrices"]

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def log_prediction(mode, model_name, values, prediction, probability):
    conn = db()
    conn.execute(
        "INSERT INTO predictions(mode,model_name,input_json,prediction,probability) VALUES(?,?,?,?,?)",
        (mode, model_name, json.dumps(values, ensure_ascii=False), int(prediction), float(probability))
    )
    conn.commit()
    conn.close()

def display_label(value):
    return "Dengue complicado" if int(value) == 1 else "Dengue no complicado"

@app.route("/")
def index():
    conn = db()
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    class_counts = conn.execute(
        "SELECT target, COUNT(*) AS total FROM patients GROUP BY target ORDER BY target"
    ).fetchall()
    pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    return render_template("index.html", metrics=METRICS, matrices=MATRICES,
        count=count, class_counts=class_counts, pred_count=pred_count,
        bp_note=bundle.get("bp_mapping",""))

@app.route("/individual", methods=["GET","POST"])
def individual():
    result = None
    if request.method == "POST":
        try:
            values = {
                "BP_AT_ADMISSION": 1 if request.form["hypotension"] == "yes" else 0,
                "PCV": float(request.form["PCV"]),
                "PLATELET_COUNT": float(request.form["PLATELET_COUNT"]),
                "HIGHEST_WBC": float(request.form["HIGHEST_WBC"]),
                "ALT": float(request.form["ALT"])
            }
            sample = pd.DataFrame([values], columns=FEATURES)
            result = {}
            for name, model in MODELS.items():
                pred = int(model.predict(sample)[0])
                prob = float(model.predict_proba(sample)[0][1])
                log_prediction("individual", name, values, pred, prob)
                result[name] = {"label": display_label(pred), "prediction": pred,
                                "probability": round(prob*100,1)}
        except Exception as e:
            flash("Revisa los valores ingresados. " + str(e), "error")
    return render_template("individual.html", result=result)

@app.route("/lote", methods=["GET","POST"])
def lote():
    table = None
    download_ready = False
    if request.method == "POST":
        file = request.files.get("batch")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Selecciona un archivo CSV.", "error")
        else:
            try:
                data = pd.read_csv(file)
                missing = [c for c in FEATURES if c not in data.columns]
                if missing:
                    raise ValueError("Faltan estas columnas: " + ", ".join(missing))
                data = data.copy()
                # The CSV must use original coding: 2 = hypotension; 1 = no hypotension.
                data["BP_AT_ADMISSION"] = (pd.to_numeric(data["BP_AT_ADMISSION"], errors="coerce") == 2).astype(int)
                for c in ["PCV","PLATELET_COUNT","HIGHEST_WBC","ALT"]:
                    data[c] = pd.to_numeric(data[c], errors="coerce")
                if data[FEATURES].isna().all(axis=1).any():
                    raise ValueError("Hay filas sin datos suficientes.")
                output = data.copy()
                for name, model in MODELS.items():
                    pred = model.predict(data[FEATURES])
                    prob = model.predict_proba(data[FEATURES])[:,1]
                    col = "PRED_LOGISTICA" if name == "Regresión logística" else "PRED_RED_NEURONAL"
                    output[col] = ["Complicado" if int(x)==1 else "No complicado" for x in pred]
                    for i in range(len(data)):
                        log_prediction("lote", name, data.iloc[i][FEATURES].to_dict(), int(pred[i]), float(prob[i]))
                csv_data = output.to_csv(index=False).encode("utf-8-sig")
                app.config["LAST_CSV"] = csv_data
                table = output.head(100).fillna("").to_dict(orient="records")
                download_ready = True
                flash(f"Lote procesado: {len(output)} registros.", "success")
            except Exception as e:
                flash("No se pudo procesar el archivo: " + str(e), "error")
    return render_template("lote.html", table=table, download_ready=download_ready)

@app.route("/descargar")
def descargar():
    content = app.config.get("LAST_CSV")
    if not content:
        flash("Primero procesa un archivo por lote.", "error")
        return redirect(url_for("lote"))
    return send_file(io.BytesIO(content), mimetype="text/csv",
        as_attachment=True, download_name="predicciones_dengue.csv")

@app.route("/historial")
def historial():
    conn = db()
    rows = conn.execute(
        "SELECT id,mode,model_name,prediction,probability,created_at FROM predictions ORDER BY id DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return render_template("historial.html", rows=rows, label=display_label)

@app.route("/health")
def health():
    return {"status":"ok","database":os.path.exists(DB_PATH),"models":list(MODELS.keys())}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
