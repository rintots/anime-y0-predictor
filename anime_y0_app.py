import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="A.N.I.M.E.モデル：視聴者数予測シミュレーター", layout="wide")
st.title("A.N.I.M.E.モデル：視聴者数予測シミュレーター")

st.markdown("""
このツールは、アニメ企画段階での要素（原作、監督、主題歌など）を入力することで、
**放送前の想定視聴者数（Y₀）**をリアルタイムで予測します。
数値はそのまま人間にとってわかりやすく、裏側で自動スケーリングされます。
""")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("作品タイトル（任意）")
    original_copies = st.number_input("原作の発行部数（例：5000000）", min_value=0)
    artist_score = st.slider("主題歌アーティストスコア（0〜10）", 0.0, 10.0, 5.0)
    studio_score = st.slider("制作スタジオスコア（0〜10）", 0.0, 10.0, 5.0)
    cast_sns_score = st.slider("キャストSNSスコア（0〜10）", 0.0, 10.0, 5.0)

with col2:
    pv_views = st.number_input("PV再生数（例：1000000）", min_value=0)
    sns_posts = st.number_input("SNS話題数（例：10000）", min_value=0)
    dist_score = st.slider("配信到達スコア（配信なし=0.0〜最大=2.0）", 0.0, 2.0, 1.0)
    tv_score = st.slider("放送局カバー係数（TVなし=0.0〜全国ゴールデン=2.0）", 0.0, 2.0, 1.0)

try:
    model = joblib.load("anime_y0_model_trained.joblib")

    # 裏側でスケーリング
    X = pd.DataFrame([{
        "主題歌スコア": artist_score,
        "sqrt_PV再生数": np.sqrt(pv_views),
        "sqrt_SNS話題数": np.sqrt(sns_posts),
        "キャストSNSスコア": cast_sns_score,
        "スタジオスコア": studio_score,
        "sqrt_原作部数": np.sqrt(original_copies),
        "配信スコア": dist_score,
        "放送スコア": tv_score
    }])

    # モデルによる予測 + 上限制限
    raw_pred = model.predict(X)[0]
    max_possible_viewers = (dist_score + tv_score) * 1_000_000
    min_viewers = 10_000

    if dist_score == 0.0 and tv_score == 0.0:
        pred = 0
    else:
        pred = int(max(min_viewers, min(raw_pred, max_possible_viewers)))

    st.success(f"### 推定視聴者数：{pred:,}人")

except Exception as e:
    st.warning("モデルがまだ読み込まれていないか、予測できません。モデルファイル（anime_y0_model_trained.joblib）を配置してください。")
    st.text(str(e))
