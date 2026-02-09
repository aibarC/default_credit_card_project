import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
PAY_N = [f"PAY_{i}" for i in range(2,7)]
PAY_N.insert(0, "PAY_0")
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]


class DeafaultFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, bill_cols, method="yeo-johnson", standardize=False):
        self.bill_cols = list(bill_cols)
        self.method = method
        self.standardize = standardize
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.pt_ = PowerTransformer(method=self.method, standardize=self.standardize)
        self.pt_.fit(X[self.bill_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in PAY_N:
            X[col] = X[col].clip(upper=3)
        X["LIMIT_BAL_LOG"]=np.log1p(X["LIMIT_BAL"])
        X["EDUCATION"]=X["EDUCATION"].replace({1:'graduate school', 2:'university', 
                                               3:'high school', 4:'others', 
                                               5:'others', 6:'others', 0:'others'})
        X['SEX']=X['SEX'].replace({1:'male', 2:'female'})
        X['MARRIAGE']=X['MARRIAGE'].replace({1:'married', 2:'single', 3:'others', 0:'others'})
        for col in PAY_AMT:
            X[col] = np.log1p(X[col])
        X[self.bill_cols]=self.pt_.transform(X[self.bill_cols])
        return X


def build_custom_pipeline(best_params: dict, random_state: int = 42, final_pipe=None) -> Pipeline:
    return Pipeline([
        ("fe", DeafaultFeatureEngineer(bill_cols=BILL_AMT, method="yeo-johnson", standardize=False)),
        ("pipe", final_pipe)
    ])