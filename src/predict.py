from pathlib import Path
import json

import joblib
import pandas as pd

MODEL_DATA_PATH = Path('artifacts') / 'model_data'
MODEL_PATH= MODEL_DATA_PATH / 'models' / 'full_custom_final_model.joblib'
THRESHOLD_PATH = MODEL_DATA_PATH  / 'thresholds.json'

_model=None
_threshold=None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def get_threshold() -> float:
    global _threshold
    if _threshold is None:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            _threshold = float(json.load(f)['threshold_recall'])
    return _threshold

def predict(raw_input: dict) -> dict:
    """
    raw_input keys expected:
    LIMIT_BAL_LOG, EDUCATION, MARRIAGE, SEX, 
    PAY_N, PAY_AMTN, BIL_AMTN
    returns:
      {
        "proba": float,
        "percent": "69.0%",
        "threshold": float,
        "pred": 0/1
      }
    """
    model = get_model()
    thr = get_threshold()

    X = pd.DataFrame([{
        "LIMIT_BAL": raw_input.get("LIMIT_BAL", 0.0),
        "EDUCATION": raw_input.get("EDUCATION", 'others'),
        "MARRIAGE": raw_input.get("MARRIAGE", 'others'),
        "SEX": raw_input.get("SEX", 1),
        "PAY_0": raw_input.get("PAY_0", 0),
        "PAY_2": raw_input.get("PAY_2", 0),
        "PAY_3": raw_input.get("PAY_3", 0),
        "PAY_4": raw_input.get("PAY_4", 0),
        "PAY_5": raw_input.get("PAY_5", 0),
        "PAY_6": raw_input.get("PAY_6", 0),
        "BILL_AMT1": raw_input.get("BILL_AMT1", 0.0),
        "BILL_AMT2": raw_input.get("BILL_AMT2", 0.0),
        "BILL_AMT3": raw_input.get("BILL_AMT3", 0.0),
        "BILL_AMT4": raw_input.get("BILL_AMT4", 0.0),
        "BILL_AMT5": raw_input.get("BILL_AMT5", 0.0),
        "BILL_AMT6": raw_input.get("BILL_AMT6", 0.0),
        "PAY_AMT1": raw_input.get("PAY_AMT1", 0.0),
        "PAY_AMT2": raw_input.get("PAY_AMT2", 0.0),
        "PAY_AMT3": raw_input.get("PAY_AMT3", 0.0),
        "PAY_AMT4": raw_input.get("PAY_AMT4", 0.0),
        "PAY_AMT5": raw_input.get("PAY_AMT5", 0.0),
        "PAY_AMT6": raw_input.get("PAY_AMT6", 0.0),
    }])
    #in our case the recall_threshold would be used
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "proba": proba,
        "percent": f"{proba * 100:.1f}%",
        'threshold': thr,
        "pred": int(proba >= thr),
    }
# def main():
#     # Example usage
#     raw_input = {
#         "PAY_0":2, 
#         "PAY_2":2, 
#         "PAY_3":3, 
#         "PAY_4":2, 
#         "PAY_5":0, 
#         "PAY_6":0,
#         "SEX":2,
#         "EDUCATION":2,
#         "MARRIAGE":2,
#         "BILL_AMT1": 161771.0,
#         "BILL_AMT2": 172632.0,
#         "BILL_AMT3": 168541.0,
#         "BILL_AMT4": 164310.0,
#         "BILL_AMT5": 162681.0,
#         "BILL_AMT6": 163005.0,
#         "PAY_AMT1": 15000.0,
#         "PAY_AMT2": 0.0,
#         "PAY_AMT3": 0.0,
#         "PAY_AMT4": 6100.0,
#         "PAY_AMT5": 12300.0,
#         "PAY_AMT6": 6100.0,
#         "LIMIT_BAL": 160000.0,
#     }
#     result = predict(raw_input)
#     print(result)
# if __name__ == "__main__":
#     main()
        