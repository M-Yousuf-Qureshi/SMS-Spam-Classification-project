import streamlit as st
import pickle
import re
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Smart Spam & Email Analyzer",
    page_icon="📧",
    layout="centered"
)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    """
    Loads the model and vectorizer. 
    Checks multiple possible paths for convenience.
    """
    # Try local root first, then the 'models' folder
    paths_to_check = [
        ("model.pkl", "vectorizer.pkl"),
        (os.path.join(os.getcwd(), "models", "naive_bayes_model.pkl"), os.path.join("models", "naive_bayes_vectorizer.pkl"))
    ]
    
    for m_path, v_path in paths_to_check:
        if os.path.exists(m_path) and os.path.exists(v_path):
            try:
                with open(m_path, "rb") as f:
                    model = pickle.load(f)
                with open(v_path, "rb") as f:
                    vectorizer = pickle.load(f)
                return model, vectorizer
            except Exception as e:
                st.error(f"Error loading files: {e}")
    
    return None, None

model, vectorizer = load_model()

# Check if model loaded successfully before proceeding
if model is None or vectorizer is None:
    st.error("❌ Model or Vectorizer files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' exist.")
    st.stop()

# ===============================
# HELPER FUNCTIONS
# ===============================

def predict_message(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    # FIX: Not all models (like LinearSVC) support predict_proba
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(text_vector)[0]
        confidence = max(prob) * 100
    else:
        # For models like LinearSVC, use decision_function or just return None
        confidence = None 

    return prediction, confidence


def check_email_authenticity(email):
    suspicious_domains = [
        "gmail-security.com", "paypal-alert.com", "amazon-support.net",
        "verify-account.org", "secure-login.io"
    ]

    if "@" not in email:
        return "❌ Invalid Email Format"

    domain = email.split("@")[-1].lower()

    if domain in suspicious_domains:
        return "🚨 Suspicious Email Domain Detected!"

    if re.search(r"\d{5,}", email): # More than 5 consecutive digits
        return "⚠ Suspicious Pattern (Too many numbers)"

    return "✅ Domain looks standard."


def check_keywords(text):
    suspicious_words = [
        "win", "free", "urgent", "lottery", "click now", 
        "claim", "offer", "prize", "congratulations", "cash"
    ]
    found = [word for word in suspicious_words if word in text.lower()]
    return found


# ===============================
# UI HEADER
# ===============================

st.title("📧 Smart Spam & Email Analyzer")
st.markdown("AI-powered Spam Detection & Email Risk Analysis")

# ===============================
# MODE SELECTION
# ===============================

mode = st.selectbox(
    "Select Analysis Mode",
    ["SMS/Email Spam Detection", "Email Authenticity Check"]
)

# ===============================
# SPAM DETECTION MODE
# ===============================

if mode == "SMS/Email Spam Detection":
    user_text = st.text_area("Enter Message Text", height=150, placeholder="Paste email or SMS content here...")

    if st.button("Analyze Message"):
        if not user_text.strip():
            st.warning("⚠ Please enter some text to analyze.")
        else:
            prediction, confidence = predict_message(user_text)

            if prediction == 1:
                conf_text = f" | Confidence: {confidence:.2f}%" if confidence else ""
                st.error(f"🚨 SPAM DETECTED{conf_text}")
            else:
                conf_text = f" | Confidence: {confidence:.2f}%" if confidence else ""
                st.success(f"✅ NOT SPAM{conf_text}")

            # Secondary Check: Keywords
            keywords = check_keywords(user_text)
            if keywords:
                st.warning(f"⚠ Suspicious Keywords Found: {', '.join(keywords)}")

# ===============================
# EMAIL AUTHENTICITY MODE
# ===============================

elif mode == "Email Authenticity Check":
    email_input = st.text_input("Enter Sender's Email Address", placeholder="e.g. support@paypal-alert.com")

    if st.button("Verify Sender"):
        if not email_input.strip():
            st.warning("⚠ Please enter an email address.")
        else:
            result = check_email_authenticity(email_input)
            if "✅" in result:
                st.success(result)
            else:
                st.error(result)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Developed with Scikit-Learn & Streamlit")
