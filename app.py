import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from config import MODEL_PATH, MODEL_DIR, DATA_PATH
from data_loader import load_dataset
from train_model import train
from mood_logic import mood_from_scores, normalize_score
from history import init_db, save_analysis, get_history

st.set_page_config(page_title="Moodify AI", page_icon="🎧", layout="wide")
init_db()

st.markdown("""
<style>
.main {background: #0b1020;}
.block-container {padding-top: 2rem; max-width: 1400px;}
[data-testid="stMetric"] {background: #151d33; padding: 18px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Moodify AI")
st.caption("Music emotion recognition • Valence & arousal • Song discovery")
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Overview", "Analyze a feature record", "Explore DEAM", "Recommendations", "History", "Model performance"])
    st.divider()
    st.caption("Dataset: DEAM song-level annotations and openSMILE descriptors.")

@st.cache_resource
def get_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

bundle = get_bundle()

if page == "Overview":
    st.subheader("Understand the emotional character of music")
    c1,c2,c3 = st.columns(3)
    c1.metric("Dataset", "DEAM")
    c2.metric("Songs", "1,802")
    c3.metric("Acoustic descriptors", "520")
    st.write("Moodify predicts two continuous emotion dimensions:")
    a,b = st.columns(2)
    a.info("**Valence** — how positive or negative the music feels.")
    b.info("**Arousal** — how calm or energetic the music feels.")
    st.warning("Train the models before using prediction pages. Use the button below to download the real DEAM-derived feature table and train models.")
    if st.button("Download dataset and train models", type="primary"):
        with st.spinner("Downloading and training. This can take a few minutes..."):
            try:
                new_bundle, df = train()
                st.cache_resource.clear()
                st.success(f"Training complete. Selected: {new_bundle['model_name']}. Dataset rows: {len(df)}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not complete training: {e}")

elif page == "Analyze a feature record":
    st.subheader("Predict emotion from an acoustic feature record")
    if bundle is None:
        st.warning("Train the models from the Overview page first.")
    else:
        st.write("This page accepts a DEAM-compatible acoustic feature row. It does not claim to predict from arbitrary MP3 audio.")
        uploaded = st.file_uploader("Upload a CSV containing one or more rows with the same feature columns", type=["csv"])
        if uploaded:
            sample = pd.read_csv(uploaded)
            absent = [c for c in bundle["features"] if c not in sample.columns]
            if absent:
                st.error(f"This CSV is missing {len(absent)} required model features. Use the DEAM feature schema.")
            else:
                preds = bundle["model"].predict(sample[bundle["features"]])
                out = pd.DataFrame(preds, columns=["Predicted valence", "Predicted arousal"])
                out["Mood"] = [mood_from_scores(v,a) for v,a in preds]
                st.dataframe(out, use_container_width=True)
        st.caption("For direct MP3 inference, the training and inference audio pipeline must use the same feature extractor and compatible training audio. That is a planned extension, not silently simulated here.")

elif page == "Explore DEAM":
    st.subheader("Dataset exploration")
    try:
        df = load_dataset()
        x,y,z = st.columns(3)
        x.metric("Songs", f"{len(df):,}")
        y.metric("Columns", f"{len(df.columns):,}")
        z.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.write("Valence and arousal are human annotation averages on a 1–9 scale.")
        st.dataframe(df[["song_id","valence_mean","arousal_mean","valence_std","arousal_std"]].head(20), use_container_width=True)
        fig = px.scatter(df, x="valence_mean", y="arousal_mean", hover_data=["song_id"],
                         title="DEAM songs in valence–arousal space")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Annotation distributions")
        dfl = df[["valence_mean","arousal_mean"]].melt(var_name="dimension", value_name="score")
        st.plotly_chart(px.histogram(dfl, x="score", color="dimension", barmode="overlay"), use_container_width=True)
    except Exception as e:
        st.info(f"Download the dataset from Overview first. Details: {e}")

elif page == "Recommendations":
    st.subheader("Find songs with a similar emotional profile")
    try:
        df = load_dataset()
        v = st.slider("Desired valence", 1.0, 9.0, 6.0, 0.1)
        a = st.slider("Desired arousal", 1.0, 9.0, 6.0, 0.1)
        n = st.slider("Number of recommendations", 3, 20, 8)
        pool = df.copy()
        pool["distance"] = ((pool["valence_mean"]-v)**2 + (pool["arousal_mean"]-a)**2)**0.5
        recs = pool.nsmallest(n, "distance")
        recs["interpreted_mood"] = [mood_from_scores(x,y) for x,y in zip(recs.valence_mean,recs.arousal_mean)]
        st.dataframe(recs[["song_id","valence_mean","arousal_mean","interpreted_mood","distance"]], use_container_width=True)
        st.caption("These are recommendations by emotional similarity among DEAM song IDs. The table does not include playable audio or guaranteed commercial streaming links.")
    except Exception as e:
        st.info(f"Download the dataset from Overview first. Details: {e}")

elif page == "History":
    st.subheader("Local analysis history")
    rows = get_history()
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Date","Filename","Valence","Arousal","Mood","Model"]), use_container_width=True)
    else:
        st.info("No saved analyses yet. History is stored locally in SQLite.")

elif page == "Model performance":
    st.subheader("Held-out test performance")
    if bundle is None:
        st.warning("Train the models from the Overview page first.")
    else:
        st.write(f"Selected model: **{bundle['model_name']}**")
        results = pd.DataFrame(bundle["results"]).T.reset_index(names="Model")
        st.dataframe(results, use_container_width=True)
        st.caption("Metrics are computed on a fixed 20% hold-out split. Lower MAE/RMSE and higher R² are generally better. The selected model is chosen by combined test RMSE for this demonstration; for a rigorous final study, use cross-validation on training data for model selection and keep the test set untouched.")
