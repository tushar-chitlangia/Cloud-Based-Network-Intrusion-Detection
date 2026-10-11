import os
import io
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("NIDS_API_URL", "http://127.0.0.1:8000")
MAX_ROWS = 5000

st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI-Based Network Intrusion Detection System")
st.caption("CSV-based network traffic classification using a trained Random Forest model")

@st.cache_data(ttl=30)
def get_model_info():
    response = requests.get(f"{API_URL}/model-info", timeout=10)
    response.raise_for_status()
    return response.json()

try:
    info = get_model_info()
    feature_names = info.get("feature_names", [])
    if len(feature_names) != 68:
        st.error("The API did not provide the expected 68-feature schema.")
        st.stop()
    st.success("Prediction API connected")
except requests.RequestException:
    st.error("Cannot connect to the prediction API. Check the backend service.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Model", info.get("model_name", "Random Forest"))
m2.metric("Required features", info["feature_count"])
m3.metric("Traffic classes", len(info["classes"]))

with st.expander("Supported traffic classes"):
    st.write(", ".join(info["classes"]))

st.divider()
st.subheader("Upload Network-Flow CSV")
st.write(
    "Upload a CSV containing one row per network flow and exactly the "
    "68 numeric features used during training. Do not include the target "
    "Label column or an index column."
)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

    st.write(f"**Uploaded rows:** {len(data):,}")
    st.write(f"**Uploaded columns:** {len(data.columns):,}")

    if data.empty:
        st.error("The CSV has no data rows.")
        st.stop()

    if len(data) > MAX_ROWS:
        st.error(f"Upload at most {MAX_ROWS:,} rows per batch.")
        st.stop()

    missing = sorted(set(feature_names) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(feature_names))

    if missing or unexpected:
        st.error("CSV columns do not match the model's required schema.")
        if missing:
            st.write("**Missing columns:**", ", ".join(missing))
        if unexpected:
            st.write("**Unexpected columns:**", ", ".join(unexpected))
        st.info(
            "Use the same feature names and definitions as the model's "
            "training pipeline. Do not rename columns arbitrarily."
        )
        st.stop()

    # Force the model's training feature order.
    data = data[feature_names].copy()

    for column in feature_names:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.isna().any().any():
        bad = data.columns[data.isna().any()].tolist()
        st.error(
            "Some feature values are missing or non-numeric. "
            "Clean these columns before uploading: " + ", ".join(bad)
        )
        st.stop()

    import numpy as np
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        st.error("The CSV contains infinite or non-finite feature values.")
        st.stop()

    st.subheader("Input preview")
    st.dataframe(data.head(10), use_container_width=True)

    if st.button("Run Batch Detection", type="primary"):
        try:
            with st.spinner(f"Classifying {len(data):,} network flows..."):
                response = requests.post(
                    f"{API_URL}/predict-batch",
                    json={"records": data.to_dict(orient="records")},
                    timeout=180,
                )

            if response.status_code != 200:
                st.error(f"API returned {response.status_code}: {response.text}")
                st.stop()

            result = response.json()
            results = pd.DataFrame(result["results"])

            output = data.copy()
            output.insert(0, "Predicted_Class", results["prediction"].values)

            if "probabilities" in results.columns:
                probability_frame = pd.json_normalize(results["probabilities"])
                probability_frame.columns = [
                    f"Probability_{column}" for column in probability_frame.columns
                ]
                output = pd.concat(
                    [output.reset_index(drop=True), probability_frame.reset_index(drop=True)],
                    axis=1,
                )

            st.session_state["nids_results"] = output
            st.session_state["nids_predictions"] = results["prediction"].value_counts()
        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API: {exc}")

if "nids_results" in st.session_state:
    output = st.session_state["nids_results"]
    counts = st.session_state["nids_predictions"]

    st.divider()
    st.subheader("Detection Results")
    total = len(output)
    suspicious = int(total - counts.get("BENIGN", 0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analyzed", f"{total:,}")
    c2.metric("BENIGN predictions", f"{int(counts.get('BENIGN', 0)):,}")
    c3.metric("Non-BENIGN predictions", f"{suspicious:,}")

    st.write("**Predicted class distribution**")
    st.bar_chart(counts)

    st.write("**Predictions and probabilities**")
    st.dataframe(output, use_container_width=True, height=400)

    st.download_button(
        "Download prediction results (CSV)",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="nids_prediction_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Predictions are model outputs, not proof of malicious activity. "
        "Results depend on input quality and the model's training data."
    )
