import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import FEATURE_NAMES, extract_features_from_timestamps
from generator import make_dataset

def train(output="models/hid_rhythm_sentinel.joblib"):
    data=make_dataset(); rows=[]
    for sid,g in data.groupby("session_id"):
        f=extract_features_from_timestamps(g.timestamp_ms)
        f["target"]=int(g.label.iloc[0]=="injection"); rows.append(f)
    table=__import__("pandas").DataFrame(rows)
    X=table[FEATURE_NAMES]; y=table.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model=RandomForestClassifier(n_estimators=250,max_depth=10,random_state=42,n_jobs=-1,class_weight="balanced")
    model.fit(Xtr,ytr)
    print("Test accuracy:", accuracy_score(yte,model.predict(Xte)))
    joblib.dump({"model":model,"feature_names":FEATURE_NAMES,
                 "thresholds":{"human_max":.40,"suspicious_max":.70}},output)

if __name__=="__main__": train()
