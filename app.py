import random
import numpy as np
import joblib
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Page Configuration
st.set_page_config(
    page_title="AI Digit Studio | By Hifza",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS for Sleek Cards, Creator Badge & Engaging UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .creator-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(79, 70, 229, 0.2), rgba(6, 182, 212, 0.2));
        border: 1px solid #38BDF8;
        color: #38BDF8;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border: 1px solid #4338CA;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        color: white;
    }
    .pred-number {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        color: #38BDF8;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .pred-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #C7D2FE;
        margin-bottom: 8px;
    }
    .waiting-box {
        border: 2px dashed #4B5563;
        border-radius: 16px;
        padding: 60px 20px;
        text-align: center;
        color: #9CA3AF;
    }
    .sidebar-creator {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border-left: 4px solid #38BDF8;
        padding: 12px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .footer-credit {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.95rem;
        padding: 20px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 3. Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("mnist_model.pkl")


model = load_model()

# Initialize Session State for History, Score & Challenge
if "history" not in st.session_state:
    st.session_state.history = []
if "challenge_digit" not in st.session_state:
    st.session_state.challenge_digit = None
if "score" not in st.session_state:
    st.session_state.score = {"correct": 0, "total": 0}


# 4. Preprocess Image to MNIST Format (28x28)
def prepare_digit(gray):
    gray = np.asarray(gray, dtype=np.float32)
    if gray.max() < 30:
        return None
    rows = np.where(gray.max(axis=1) > 30)[0]
    cols = np.where(gray.max(axis=0) > 30)[0]
    digit = gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = np.array(
        Image.fromarray(digit.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float32,
    )
    box = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    box[top : top + new_h, left : left + new_w] = digit
    total = box.sum()
    if total > 0:
        r, c = np.indices(box.shape)
        dy = int(round(14 - (r * box).sum() / total))
        dx = int(round(14 - (c * box).sum() / total))
        box = np.roll(np.roll(box, dy, axis=0), dx, axis=1)
    return (box / 255.0).reshape(1, 784)


# 5. Sidebar Controls & Creator Credit
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-creator">
            <div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:1px;">Model & App Creator</div>
            <div style="font-size:1.25rem; font-weight:700; color:#F8FAFC;">👩‍💻 Hifza</div>
            <div style="font-size:0.85rem; color:#38BDF8;">MNIST Handwritten Digit Classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("🎛️ Studio Controls")
    stroke_width = st.slider("🖌️ Brush Thickness", min_value=10, max_value=30, value=18)

    st.divider()
    st.header("🎮 AI Challenge Mode")
    st.write("Test Hifza's AI model! Can it recognize the target digit you draw?")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🎲 New Target", use_container_width=True):
            st.session_state.challenge_digit = random.randint(0, 9)
    with col_btn2:
        if st.button("❌ Clear Mode", use_container_width=True):
            st.session_state.challenge_digit = None

    if st.session_state.challenge_digit is not None:
        st.info(f"🎯 **Target Digit:** Draw a **{st.session_state.challenge_digit}**!")

    st.divider()
    st.header("📊 Session Accuracy")
    tot = st.session_state.score["total"]
    corr = st.session_state.score["correct"]
    acc = (corr / tot * 100) if tot > 0 else 0.0
    st.metric("Verified Accuracy", f"{acc:.0f}%", f"{corr}/{tot} verified")

    if st.button("🗑️ Reset History & Stats", use_container_width=True):
        st.session_state.history = []
        st.session_state.score = {"correct": 0, "total": 0}
        st.rerun()


# 6. Main Header with Hifza's Name Badge
st.markdown('<div class="main-header">🧠 AI Handwritten Digit Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="creator-badge">✨ Model Built & Trained by Hifza</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Draw any number (0–9) or upload an image to test the MNIST neural model trained by <b>Hifza</b>.</div>',
    unsafe_allow_html=True,
)

# 7. Split Workspace: Left (Input) | Right (AI Analysis)
left_col, right_col = st.columns([1.05, 1.25], gap="large")
gray = None

with left_col:
    st.subheader("1️⃣ Input Canvas")
    tab_draw, tab_upload = st.tabs(["✏️ Draw Digit", "📤 Upload Image"])

    with tab_draw:
        canvas = st_canvas(
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=300,
            height=300,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        st.caption("💡 Tip: Draw large and centered. Click the trash icon below the canvas to clear.")
        if canvas.image_data is not None:
            gray = np.array(canvas.image_data)[:, :, 0]

    with tab_upload:
        file = st.file_uploader("Upload a clear PNG or JPG of a single digit", type=["png", "jpg", "jpeg"])
        if file is not None:
            img = np.array(Image.open(file).convert("L"), dtype=np.float32)
            if img.mean() > 127:
                img = 255 - img
            gray = img
            st.image(file, caption="Uploaded Image", width=220)


# 8. Real-Time Prediction & Interactive Results
with right_col:
    st.subheader("2️⃣ AI Vision & Prediction")
    x = prepare_digit(gray) if gray is not None else None

    if x is None:
        st.markdown(
            """
            <div class="waiting-box">
                <h3>✍️ Waiting for your drawing...</h3>
                <p>Draw a digit on the left canvas or upload an image to test <b>Hifza's AI model</b>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        probs = model.predict_proba(x)[0]
        classes = [int(c) for c in model.classes_]
        top_idx = int(probs.argmax())
        digit = classes[top_idx]
        confidence = float(probs[top_idx])

        # Check Challenge Mode
        if st.session_state.challenge_digit is not None:
            if digit == st.session_state.challenge_digit and confidence > 0.60:
                st.success(f"🎉 Challenge Passed! Hifza's model recognized **{digit}** with {confidence:.0%} confidence!")
            else:
                st.warning(f"🎯 Target is **{st.session_state.challenge_digit}** — Model currently sees **{digit}** ({confidence:.0%}).")

        # Top Card + Model Input View
        card_col, eye_col = st.columns([1.3, 1])
        with card_col:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="pred-label">Hifza's Model Prediction</div>
                    <div class="pred-number">{digit}</div>
                    <div style="margin-top:8px; font-size:1.1rem; color:#A5B4FC;">
                        Confidence: <b>{confidence:.1%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)

        with eye_col:
            st.image(
                x.reshape(28, 28),
                caption="Normalized 28×28 Input",
                width=165,
                clamp=True,
            )

        # Top 3 Runner-up Guesses
        sorted_indices = np.argsort(probs)[::-1]
        st.markdown("#### 🏆 Top 3 Model Guesses")
        t1, t2, t3 = st.columns(3)
        for col_box, idx in zip([t1, t2, t3], sorted_indices[:3]):
            col_box.metric(f"Digit {classes[idx]}", f"{probs[idx]:.1%}")

        # Probability Distribution Chart
        with st.expander("📊 Full Probability Breakdown (0–9)", expanded=True):
            df_probs = pd.DataFrame(
                {"Digit": [str(c) for c in classes], "Probability": [float(p) * 100 for p in probs]}
            ).set_index("Digit")
            st.bar_chart(df_probs, height=200)

        # User Feedback Buttons
        st.markdown("#### Did Hifza's AI guess right?")
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("👍 Yes, Correct!", use_container_width=True):
                st.session_state.score["correct"] += 1
                st.session_state.score["total"] += 1
                st.session_state.history.insert(0, {"img": x.reshape(28, 28), "pred": digit, "conf": confidence, "status": "✅"})
                st.toast("Awesome! Recorded as correct.", icon="🎉")
        with fb2:
            if st.button("👎 No, Wrong", use_container_width=True):
                st.session_state.score["total"] += 1
                st.session_state.history.insert(0, {"img": x.reshape(28, 28), "pred": digit, "conf": confidence, "status": "❌"})
                st.toast("Feedback saved! Try drawing thicker or more centered.", icon="📝")


# 9. Recent Predictions Gallery
if st.session_state.history:
    st.divider()
    st.subheader("🕒 Recent Verified Attempts")
    hist_cols = st.columns(min(len(st.session_state.history), 6))
    for i, item in enumerate(st.session_state.history[:6]):
        with hist_cols[i]:
            st.image(item["img"], width=85, clamp=True)
            st.caption(f"{item['status']} Pred: **{item['pred']}** ({item['conf']:.0%})")

# 10. Footer Credit
st.divider()
st.markdown(
    '<div class="footer-credit">Developed & Trained with ❤️ by <b>Hifza</b> | Powered by MNIST & Streamlit</div>',
    unsafe_allow_html=True,
)