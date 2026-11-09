import os, json, sqlite3, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE=os.path.dirname(os.path.abspath(__file__))
EXCEL=os.path.join(BASE,"MASTER_CHART_F1000Research.xlsx")
DB=os.path.join(BASE,"dengue.db")
MODEL=os.path.join(BASE,"models","modelos_entrenados.joblib")
FEATURES=["BP_AT_ADMISSION","PCV","PLATELET_COUNT","HIGHEST_WBC","ALT"]

df=pd.read_excel(EXCEL)
df.columns=[str(c).strip().replace(" ","_").replace(":","").replace("/","_") for c in df.columns]
target=df.columns[0]
data=df.dropna(subset=[target]+FEATURES).copy()
X=data[FEATURES].copy()
X["BP_AT_ADMISSION"]=(X["BP_AT_ADMISSION"]==2).astype(int)
y=(data[target]==2).astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
models={
 "Regresión logística":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42))]),
 "Red neuronal artificial":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",MLPClassifier(hidden_layer_sizes=(16,8),activation="relu",solver="lbfgs",alpha=.01,max_iter=3000,random_state=42))])
}
metrics,matrices={},{}
for name,model in models.items():
 model.fit(Xtr,ytr); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
 metrics[name]={"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),
 "recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),
 "roc_auc":float(roc_auc_score(yte,prob)),"test_records":int(len(yte))}
 matrices[name]=confusion_matrix(yte,pred).tolist()
os.makedirs(os.path.join(BASE,"models"),exist_ok=True)
joblib.dump({"models":models,"features":FEATURES,"target":target,"metrics":metrics,"matrices":matrices,
 "bp_mapping":"Excel code 2 = hypotension (cross-checked against article Table 1)"},MODEL)
# Refresh DB's patient and metric tables; keep prediction history intact.
con=sqlite3.connect(DB)
con.execute("DELETE FROM patients"); con.execute("DELETE FROM model_metrics")
for idx,row in data.iterrows():
 original={k:(None if pd.isna(v) else (v.item() if hasattr(v,"item") else v)) for k,v in row.to_dict().items()}
 feat={k:(None if pd.isna(v) else (v.item() if hasattr(v,"item") else v)) for k,v in X.loc[idx].to_dict().items()}
 con.execute("INSERT INTO patients(original_row,target,features_json,original_json) VALUES(?,?,?,?)",
 (int(idx)+2,int(y.loc[idx]),json.dumps(feat),json.dumps(original)))
for name,m in metrics.items():
 con.execute("INSERT INTO model_metrics(model_name,accuracy,precision,recall,f1,roc_auc,test_records,confusion_matrix_json) VALUES(?,?,?,?,?,?,?,?)",
 (name,m["accuracy"],m["precision"],m["recall"],m["f1"],m["roc_auc"],m["test_records"],json.dumps(matrices[name])))
con.commit(); con.close()
with open(os.path.join(BASE,"informe_entrenamiento.json"),"w",encoding="utf-8") as f:
 json.dump({"metrics":metrics,"confusion_matrices":matrices,"rows_used":len(data),"features":FEATURES},f,ensure_ascii=False,indent=2)
print(json.dumps(metrics,ensure_ascii=False,indent=2))
