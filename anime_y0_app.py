import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="A.N.I.M.E.モデル：視聴者数予測シミュレーター", layout="wide")
st.title("A.N.I.M.E.モデル：視聴者数予測シミュレーター")

st.markdown("""
このツールは、アニメ企画段階での要素（原作、監督、主題歌など）を入力することで、**放送前の想定視聴者数（Y₀）**をリアルタイムで予測します。
""")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("作品タイトル（任意）")
    original_copies = st.number_input("原作部数（例：5000000）", min_value=0)
    artist_score = st.slider("主題歌アーティストスコア（0∼10）", 0.0, 10.0, 5.0)
    studio_score = st.slider("制作スタジオスコア（0∼10）", 0.0, 10.0, 5.0)

with col2:
    pv_views = st.number_input("PV再生数（放送前）", min_value=0)
    sns_posts = st.number_input("SNS話題数（放送前）", min_value=0)
    dist_score = st.slider("配信到達スコア（配信なし=0.0, FOD独占=0.5, dアニメ単独=0.8, Netflix=1.7, 最大=2.0）", 0.0, 2.0, 1.0)
    tv_score = st.slider("放送局カバー係数（TVなし=0.0, ローカル深夜=0.5, 関東ローカル=1.0, 全国ゴールデン=2.0）", 0.0, 2.0, 1.0)

try:
    model = joblib.load("anime_y0_model.joblib")

    X = pd.DataFrame([{
        "主題歌スコア": artist_score,
        "sqrt_PV再生数": np.sqrt(pv_views),
        "sqrt_SNS話題数": np.sqrt(sns_posts),
        "スタジオスコア": studio_score,
        "sqrt_原作部数": np.sqrt(original_copies),
        "配信到達スコア": dist_score,
        "放送局カバー係数": tv_score
    }])

    if dist_score == 0.0 and tv_score == 0.0:
        pred = 0  # 配信も放送もないなら視聴者ゼロ
    else:
        pred = int(model.predict(X)[0])

    st.success(f"### 推定視聴者数：{pred:,}人")

except Exception as e:
    st.warning("モデルがまだ読み込まれていないか、予測できません。学習済みモデルを配置してください。")
    st.text(str(e))
