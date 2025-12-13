import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

df=pd.read_csv("data/raw_data.csv")
df=df.drop_duplicates()

#unwanted columns remove 
df=df.drop(columns=["patient_id",
    "los_hours",
    "baseline_risk_score",
    "deterioration_event",
    "deterioration_within_12h_from_admission",
    "deterioration_next_12h"])
#target column logic 
df["deterioration_next_1h"] = (
    (df["deterioration_hour"] - df["hour_from_admission"]).between(0,1)
).astype(int)

#print((df["deterioration_next_1h"]==1).sum())
#removing the deterioration_hour column bcz our work done from generating target column so removing this column is necessary due to model not cheats
df=df.drop(columns=["deterioration_hour"])
print("this is the columns that we need to wnat as input before feature engineering ",df.drop(columns=["deterioration_next_1h"]).columns)
#eda 
plt.scatter(df["heart_rate"],df["deterioration_next_1h"])
plt.xlabel("heart  rate ")
plt.ylabel("deterioration probability")
plt.show()
plt.scatter(df["spo2_pct"],df["deterioration_next_1h"])
plt.xlabel("spo2 percentage %")
plt.ylabel("deteriorTRION probabilty ")
plt.show()
plt.hist(df["deterioration_next_1h"],bins=10,color="purple",edgecolor="pink")
plt.show()
#key insgihts _
#the patient who have hr bw 50 to  142 shows most detrioation probabliyy in their next 1 hr
#the patients who have spo2 pct 74 to 97 shows most detrioation probabliyy in their next 1 hr

#now feature engineering
df["MAP"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
df["shock_index"] = df["heart_rate"] / df["systolic_bp"]
df["spo2_gap"] = 100 - df["spo2_pct"]
    # --------------------------
    # 4. LAB ENGINEERED FEATURES
    # --------------------------
df["inflammation_index"] = df["wbc_count"] * df["crp_level"]
df["organ_stress"] = df["lactate"] + df["creatinine"]
df["anemia_flag"] = (df["hemoglobin"] < 12).astype(int)
df["kidney_risk"] = (df["creatinine"] > 1.2).astype(int)
df["high_lactate_flag"] = (df["lactate"] > 2).astype(int)

    # --------------------------
    # 5. OXYGEN THERAPY FEATURES
    # --------------------------
    # Map device strings to numeric scale
device_map = {
        "none": 0,
        "room_air": 0,
        "nasal_cannula": 1,
        "mask": 2,
        "non_rebreather": 2,
        "bipap": 3,
        "cpap": 3,
        "ventilator": 4
    }
df["oxygen_device_encoded"] = df["oxygen_device"].str.lower().map(device_map)
df["oxygen_device_encoded"] = df["oxygen_device_encoded"].fillna(0)

df["oxygen_need_score"] = df["oxygen_flow"].fillna(0) * df["oxygen_device_encoded"]

    # --------------------------
    # 6. DEMOGRAPHIC FEATURES
    # --------------------------
df["gender_encoded"] = df["gender"].replace({"M": 1, "F": 0}).fillna(0)
df["age_bucket"] = pd.cut(df["age"], bins=[0, 30, 50, 70, 120], labels=[0,1,2,3]).astype(float)

    # --------------------------
    # 7. ADMISSION FEATURES
    # --------------------------
df["admission_type_encoded"] = df["admission_type"].astype("category").cat.codes

    # --------------------------
    # 8. STABILITY INDEX
    # --------------------------
df["early_stability_score"] = (
        df["MAP"].fillna(0) +
        df["spo2_pct"].fillna(0) +
        (100 - df["heart_rate"].fillna(0))
    )
df=df.replace([np.inf,-np.inf],np.nan)


#final model trainning

sc1=StandardScaler()
x=df.drop(columns=["deterioration_next_1h","oxygen_device","gender","admission_type"]).values

y=df["deterioration_next_1h"].values
#spltting train and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_train=sc1.fit_transform(x_train)
x_test=sc1.transform(x_test)

logreg = LogisticRegression(
    max_iter=3000,
    solver="liblinear",
    penalty="l1",
    class_weight="balanced"
)


# 2. Train model
logreg.fit(x_train, y_train)

# 3. Predictions
y_pred_log = logreg.predict(x_test)
y_prob_log = logreg.predict_proba(x_test)[:,1]

# 4. Evaluation
print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", logreg.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
print("logistic trainnning complted ")
"""
"""
print("Reached RF block")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
y_prob_rf = rf.predict_proba(x_test)[:,1]

print("rf  trainnig completed ")
print("=== RANDOM FOREST ===")
print("Accuracy:", rf.score(x_test, y_test))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# get probabilities
log_prob = logreg.predict_proba(x_test)[:, 1]
rf_prob = rf.predict_proba(x_test)[:, 1]



threshold =  0.260
#final_prob = 0.25* log_prob + 0.75 * rf_prob


final_prob = 0.25*log_prob+0.75*rf_prob
final_pred = (final_prob >= threshold).astype(int)




print("\n=== ENSEMBLE MODEL  of lr and rf  ===")
print("Accuracy:", accuracy_score(y_test, final_pred))
print(confusion_matrix(y_test, final_pred))
print(classification_report(y_test, final_pred))
print("ROC-AUC:", roc_auc_score(y_test, final_pred))




importance_lr = abs(logreg.coef_).mean(axis=0)
importance_lr=np.array(importance_lr)#feature sequence wise importance of features
features=df.drop(columns=["deterioration_next_1h","oxygen_device","gender","admission_type"]).columns#sequnce wise features 
indices2=np.argsort(importance_lr)[::-1]
imp_features_for_lr=pd.DataFrame({
    "features ":features[indices2],
    "importnce":importance_lr[indices2]
})
importances_rf = rf.feature_importances_   # RF ka trained model
indices = np.argsort(importances_rf)[::-1]       # sort descending
imp_features_for_rf = pd.DataFrame({
    "feature": features[indices],
    "importance": importances_rf[indices]
})

print("top 20 features for rf",imp_features_for_rf.head(20))  
print("top 20 faetures for lr ",imp_features_for_lr.head(20))
imp_features_for_rf.head(20).to_csv("top_20_features_for_rf")
imp_features_for_lr.head(20).to_csv("top_20_features_for_lr")

#saving models and scalers and faetures  for backend use

joblib.dump(logreg,"logistic_model.pkl")
joblib.dump(rf,"random_forest_model.pkl")
joblib.dump(sc1,"scaler.pkl")
joblib.dump(features.to_list(), "feature_columns.pkl")
print("saved all models 😤")


print("Training features count:", len(features))
print(features)
# SAVE ADMISSION TYPE CATEGORY MAP (VERY IMPORTANT)
admission_map = dict(
    zip(
        df["admission_type"].astype("category").cat.categories,
        df["admission_type"].astype("category").cat.codes
    )
)
joblib.dump(admission_map, "admission_map.pkl")
print("Saved admission_type encoding map:", admission_map)
print("festure sequences in trainning = ",features)

