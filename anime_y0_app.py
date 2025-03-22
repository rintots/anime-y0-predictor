
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("A.N.I.M.E.モデル：視聴者数予測シミュレーター")
st.markdown("""
このツールは、アニメ企画段階での要素（原作、監督、主題歌など）を入力することで、
**放送前の想定視聴者数（Y₀）** をリアルタイムで予測します。
""")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("作品タイトル（任意）")
    original_circulation = st.number_input("原作部数（例：5000000）", min_value=0)
    op_score = st.slider("主題歌アーティストスコア（0〜10）", 0.0, 10.0, 5.0)
    studio_score = st.slider("制作スタジオスコア（0〜10）", 0.0, 10.0, 5.0)

with col2:
    pv_views = st.number_input("PV再生数（放送前）", min_value=0)
    sns_mentions = st.number_input("SNS話題数（放送前）", min_value=0)
    platform_score = st.slider("配信到達スコア（例：FOD独占=0.8、複数=1.5）", 0.5, 2.0, 1.0)
    tv_score = st.slider("放送局カバー係数（関東ローカル=1.0、全国放送=1.5）", 0.5, 2.0, 1.0)

def preprocess_input():
    return pd.DataFrame([{
        "主題歌スコア": op_score,
        "sqrt_PV再生数": np.sqrt(pv_views),
        "sqrt_SNS話題数": np.sqrt(sns_mentions),
        "スタジオスコア": studio_score,
        "sqrt_原作部数": np.sqrt(original_circulation),
        "配信到達スコア": platform_score,
        "放送局カバー係数": tv_score
    }])

try:
    model = joblib.load("anime_y0_model.joblib")
    input_df = preprocess_input()
    pred_y0 = model.predict(input_df)[0]
    st.success(f"\n\n### 推定視聴者数：**{int(pred_y0):,}人**")
except Exception as e:
    st.warning("モデルがまだ読み込まれていないか、予測できません。学習済みモデルを配置してください。")
    st.text(str(e))
