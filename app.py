import streamlit as st

from src.predict import predict

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")

st.title("Credit Default Predictor")
st.write("Введите данные клиента — получишь вероятность дефолта.")

st.subheader("Basic info")

c1, c2, c3 = st.columns(3)
with c1:
    limit_bal = st.number_input("LIMIT_BAL", min_value=0.0, value=200000.0, step=1000.0)
    age = st.number_input("AGE", min_value=18, max_value=100, value=30, step=1)
with c2:
    sex = st.selectbox("SEX", options=[1, 2], format_func=lambda x: "male (1)" if x == 1 else "female (2)")
    education = st.selectbox("EDUCATION", options=[0, 1, 2, 3, 4, 5, 6],
                             format_func=lambda x: {
                                 0:"others(0)",1:"grad school(1)",2:"university(2)",3:"high school(3)",
                                 4:"others(4)",5:"others(5)",6:"others(6)"
                             }[x])
with c3:
    marriage = st.selectbox("MARRIAGE", options=[0, 1, 2, 3],
                            format_func=lambda x: {0:"others(0)",1:"married(1)",2:"single(2)",3:"others(3)"}[x])

# ---- Repayment status ----
st.subheader("Repayment status (PAY_*)")
pay_cols = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
pay_vals = {}
cols = st.columns(3)
for i, col in enumerate(pay_cols):
    with cols[i % 3]:
        pay_vals[col] = st.selectbox(
            col,
            options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            index=2  # 0 by default
        )

# ---- Bill amounts ----
st.subheader("Bill amounts (BILL_AMT*)")
bill_vals = {}
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
cols = st.columns(3)
for i, col in enumerate(bill_cols):
    with cols[i % 3]:
        bill_vals[col] = st.number_input(col, value=0.0, step=100.0)

# ---- Payment amounts ----
st.subheader("Payment amounts (PAY_AMT*)")
pay_amt_vals = {}
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
cols = st.columns(3)
for i, col in enumerate(pay_amt_cols):
    with cols[i % 3]:
        pay_amt_vals[col] = st.number_input(col, value=0.0, step=100.0)

# ---- Build raw_input for your predict() ----
raw_input = {
    "LIMIT_BAL": float(limit_bal),
    "AGE": int(age),
    "SEX": int(sex),
    "EDUCATION": int(education),
    "MARRIAGE": int(marriage),
    **{k: int(v) for k, v in pay_vals.items()},
    **{k: float(v) for k, v in bill_vals.items()},
    **{k: float(v) for k, v in pay_amt_vals.items()},
}

if st.button("Predict"):
    out = predict(raw_input)

    st.subheader("Result")
    st.write(f"**Default probability:** {out['percent']}")
    st.write(f"**Threshold (recall-optimized):** {out['threshold']:.4f}")

    label = "⚠️ Likely default" if out["pred"] == 1 else "✅ Not default"
    st.write(f"**Prediction:** {label}")
