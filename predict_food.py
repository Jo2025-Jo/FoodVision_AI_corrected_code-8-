
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="FoodVision AI | Intelligent Food Recognition",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "food_svm.joblib"

# --------------------------------------------------
# PROFESSIONAL CSS THEME
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d1b;
    --panel: #111a2e;
    --panel-light: #17233b;
    --border: rgba(148, 163, 184, 0.17);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --cyan: #22d3ee;
    --violet: #a78bfa;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, rgba(99,102,241,.15), transparent 38%),
        radial-gradient(ellipse at 95% 15%, rgba(6,182,212,.10), transparent 35%),
        var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: rgba(8,13,27,.8);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10182b, #0b1120);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f8fafc !important;
    letter-spacing: -0.6px;
}

p, label, .stMarkdown {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #182342 0%, #17233d 52%, #122b40 100%);
    border: 1px solid rgba(129,140,248,.27);
    border-radius: 24px;
    padding: 30px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 45px rgba(0,0,0,.18);
}

.hero h1 {
    font-size: clamp(28px, 3vw, 42px);
    font-weight: 800;
    margin: 8px 0;
    background: linear-gradient(90deg, #ffffff, #a5f3fc, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #aebed5;
    font-size: 14px;
    line-height: 1.7;
}

.eyebrow {
    color: #67e8f9;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #f1f5f9;
    margin: 18px 0 14px 0;
}

.metric-card {
    background: linear-gradient(145deg, #15213a, #10192c);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}

.metric-label {
    font-size: 12px;
    color: #9aabc4;
    font-weight: 600;
}

.metric-value {
    font-size: 28px;
    color: #f8fafc;
    font-weight: 800;
    margin: 9px 0 4px 0;
}

.metric-note {
    font-size: 11px;
    color: #67e8f9;
}

.panel {
    background: linear-gradient(145deg, rgba(22,34,57,.96), rgba(15,24,42,.96));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
}

.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    color: #a7f3d0;
    background: rgba(16,185,129,.12);
    border: 1px solid rgba(52,211,153,.25);
}

.small-muted {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.7;
}

.stButton > button {
    border: 1px solid rgba(103,232,249,.3);
    border-radius: 12px;
    background: linear-gradient(100deg, #6366f1, #0891b2);
    color: white;
    font-weight: 700;
    padding: 10px 18px;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: #67e8f9;
    box-shadow: 0 0 22px rgba(34,211,238,.2);
    transform: translateY(-1px);
    color: white;
}

[data-testid="stFileUploader"] {
    background: rgba(22,34,57,.75);
    border: 1px dashed rgba(103,232,249,.4);
    border-radius: 16px;
    padding: 14px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] button {
    color: #e2e8f0;
}

[data-testid="stImage"] img {
    border-radius: 13px;
}

div[data-testid="stProgressBar"] > div {
    background-color: #26344d;
    border-radius: 20px;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #818cf8, #22d3ee);
    border-radius: 20px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

hr {
    border-color: rgba(148,163,184,.14);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_features(image):
    """Must match the feature extraction used for SVM training."""
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (64, 64))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 8, 8], [0, 180, 0, 256, 0, 256]
    )
    histogram = cv2.normalize(histogram, histogram).flatten()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (16, 16)).flatten() / 255.0

    return np.concatenate([histogram, gray_small]).reshape(1, -1)


def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 4px 24px 4px;">
        <div style="font-size:32px;">🍱</div>
        <div style="font-size:22px;font-weight:800;color:#f8fafc;">
            FoodVision <span style="color:#67e8f9;">AI</span>
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-top:5px;">
            INTELLIGENT FOOD ANALYTICS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Food Recognition",
            "Image Preprocessing",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    if MODEL_PATH.exists():
        st.markdown(
            '<span class="status-pill">● SVM MODEL READY</span>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Model not trained yet")

    st.markdown("""
    <div style="margin-top:28px;font-size:11px;color:#718096;line-height:1.8;">
        COMPUTER VISION PROJECT<br>
        Food-101 Prototype<br>
        <span style="color:#67e8f9;">Build • Analyse • Recognise</span>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = None
model_error = None

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as error:
        model_error = str(error)


# --------------------------------------------------
# HERO
# --------------------------------------------------
def show_hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">AI • COMPUTER VISION • FOOD ANALYTICS</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <span class="status-pill">● FOODVISION WORKSPACE</span>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
if page == "Overview":
    show_hero(
        "See Food Through AI.",
        "An intelligent workspace for exploring food images, "
        "processing visual data, and analysing machine-learning predictions."
    )

    categories = list(model.classes_) if model is not None else []

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "MODEL STATUS",
            "READY" if model is not None else "NOT READY",
            "Saved SVM model" if model is not None else "Train the model first"
        )

    with c2:
        metric_card(
            "SUPPORTED CLASSES",
            str(len(categories)),
            "Categories used during training"
        )

    with c3:
        metric_card(
            "FEATURE ENGINE",
            "HSV + GRAY",
            "Colour histogram and grayscale features"
        )

    st.markdown('<div class="section-title">Explore your workspace</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <div style="font-size:28px;">🔎</div>
            <h3>Food Recognition</h3>
            <p class="small-muted">
                Upload a food photo and view the model's leading
                category predictions with their scores.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Food Recognition →", key="open_recognition",
                     use_container_width=True):
            st.session_state["next_page"] = "Food Recognition"
            st.rerun()

    with right:
        st.markdown("""
        <div class="panel">
            <div style="font-size:28px;">🧪</div>
            <h3>Image Preprocessing</h3>
            <p class="small-muted">
                Resize images, reduce noise, and improve contrast
                before computer-vision analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Preprocessing →", key="open_preprocessing",
                     use_container_width=True):
            st.session_state["next_page"] = "Image Preprocessing"
            st.rerun()

    st.markdown('<div class="section-title">Current model information</div>',
                unsafe_allow_html=True)

    if model is not None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write("**Trained categories**")
        st.write(" · ".join(name.replace("_", " ").title()
                            for name in categories))
        st.caption(
            "This prototype uses HSV colour histograms and grayscale "
            "image features with an SVM classifier."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    elif model_error:
        st.error(f"Could not load the saved model: {model_error}")
    else:
        st.warning(
            "No trained model found. Run `py train_model.py` first."
        )


# --------------------------------------------------
# FOOD RECOGNITION
# --------------------------------------------------
elif page == "Food Recognition":
    show_hero(
        "Food Recognition Studio",
        "Upload an image and let your trained baseline model analyse "
        "the visual features."
    )

    if model is None:
        st.error("The SVM model could not be loaded.")
        st.code("py train_model.py")
        if model_error:
            st.caption(model_error)

    else:
        st.markdown('<div class="section-title">01 / Upload your image</div>',
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG food image",
            type=["jpg", "jpeg", "png"],
            key="recognition_upload"
        )

        if uploaded_file is None:
            st.markdown("""
            <div class="panel" style="text-align:center;padding:32px;">
                <div style="font-size:48px;">🍕 🍔 🍣</div>
                <h3>Ready to analyse your food image?</h3>
                <p class="small-muted">
                    Upload a clear food photograph using the uploader above.
                    The model supports only its trained categories.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                features = extract_features(image)

                scores = model.predict_proba(features)[0]
                class_names = model.classes_
                order = np.argsort(scores)[::-1][:min(3, len(scores))]

                st.markdown(
                    '<div class="section-title">02 / Visual analysis</div>',
                    unsafe_allow_html=True
                )

                image_col, result_col = st.columns([1.05, 1])

                with image_col:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.subheader("Uploaded photograph")
                    st.image(image, width="stretch")
                    st.caption(
                        f"Image dimensions: {image.width} × {image.height} px"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with result_col:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.subheader("🏆 Top predictions")
                    st.caption("Ranked by the model's output score")

                    for rank, index in enumerate(order, start=1):
                        food_name = str(class_names[index]).replace(
                            "_", " "
                        ).title()
                        score = float(scores[index])

                        st.markdown(
                            f"**{rank:02d}  ·  {food_name}**"
                        )
                        st.progress(score)
                        st.caption(f"Model score: {score:.1%}")

                    st.markdown('</div>', unsafe_allow_html=True)

                best_index = int(order[0])
                best_food = str(class_names[best_index]).replace(
                    "_", " "
                ).title()

                st.markdown('<div class="section-title">03 / Analysis summary</div>',
                            unsafe_allow_html=True)

                r1, r2 = st.columns(2)

                with r1:
                    metric_card(
                        "LEADING PREDICTION",
                        best_food,
                        "Highest output score for this image"
                    )

                with r2:
                    metric_card(
                        "CANDIDATES SHOWN",
                        str(len(order)),
                        "Top ranked classes"
                    )

                st.info(
                    "Interpretation: these scores come from the SVM model. "
                    "They are not guaranteed probabilities of correctness. "
                    "A prediction may be wrong, especially with the current "
                    "27% baseline test accuracy. Results are limited to "
                    "the food categories used in training."
                )

            except Exception as error:
                st.error("Image analysis failed.")
                st.exception(error)


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
elif page == "Image Preprocessing":
    show_hero(
        "Image Processing Lab",
        "Prepare food photographs for visual analysis using standard "
        "computer-vision operations."
    )

    uploaded_file = st.file_uploader(
        "Upload an image to process",
        type=["jpg", "jpeg", "png"],
        key="preprocessing_upload"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            original = np.array(image)

            resized = cv2.resize(original, (224, 224))
            denoised = cv2.GaussianBlur(resized, (3, 3), 0)

            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            enhanced_l = clahe.apply(l_channel)

            enhanced_lab = cv2.merge(
                (enhanced_l, a_channel, b_channel)
            )
            enhanced = cv2.cvtColor(
                enhanced_lab, cv2.COLOR_LAB2RGB
            )

            st.markdown(
                '<div class="section-title">Processing results</div>',
                unsafe_allow_html=True
            )

            images = [
                ("Original Image", original),
                ("Resized · 224 × 224", resized),
                ("Noise Reduced", denoised),
                ("Contrast Enhanced", enhanced)
            ]

            for start in range(0, len(images), 2):
                columns = st.columns(2)

                for column, (title, array) in zip(
                    columns, images[start:start + 2]
                ):
                    with column:
                        st.markdown(
                            '<div class="panel">',
                            unsafe_allow_html=True
                        )
                        st.subheader(title)
                        st.image(array, width="stretch")
                        st.markdown('</div>', unsafe_allow_html=True)

            st.success("All four image views generated successfully.")

        except Exception as error:
            st.error("Could not process this image.")
            st.exception(error)

    else:
        st.info("Upload a food image to start the processing lab.")


# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif page == "About Project":
    show_hero(
        "About FoodVision AI",
        "A computer-vision project for food image analysis and automated "
        "dish classification."
    )

    st.markdown("""
    <div class="panel">
        <h3>🎯 Project objective</h3>
        <p class="small-muted">
            To build a visual recognition framework that analyses food
            photographs and predicts a food category using image features
            and machine-learning techniques.
        </p>
        <hr>
        <h3>⚙️ Current technology stack</h3>
        <p class="small-muted">
            Python · Streamlit · OpenCV · NumPy · Scikit-learn · SVM
        </p>
        <hr>
        <h3>🧠 Current classification approach</h3>
        <p class="small-muted">
            HSV colour histograms and grayscale pixel features are
            extracted from images. An SVM classifier uses those features
            to rank the trained food categories.
        </p>
        <hr>
        <h3>📌 Current limitation</h3>
        <p class="small-muted">
            The current model is a small five-category prototype. Its
            measured test accuracy is 27% on 75 test images. This is a
            baseline result, not a production-ready food recognition system.
            Further feature engineering and deep-learning experiments are
            needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#718096;font-size:11px;padding:12px;">
    FOODVISION AI · FOOD IMAGE CLASSIFICATION & CULINARY IMAGE ANALYSIS
    <br>Academic computer-vision prototype
</div>
""", unsafe_allow_html=True)
