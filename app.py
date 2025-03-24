
import streamlit as st
import google.generativeai as genai
import joblib
import numpy as np
import os
import base64

# =================== CSS for Styling & Animations =================== #
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, rgba(10, 10, 10, 0.8), rgba(40, 40, 40, 0.9)),
                        url("img4.avif") no-repeat center center fixed;
            background-size: cover;
            color: white;
        }
        .stTabs [role="tab"] {
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 10px;
            font-size: 18px;
            transition: all 0.3s ease-in-out;
        }
        .stTabs [role="tab"]:hover {
            background-color: rgba(255, 255, 255, 0.5);
        }
        .stButton>button {
            border-radius: 10px;
            background: #ff9800;
            color: white;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background: #e65100;
        }
        .stTextInput, .stNumberInput, .stSelectbox, .stMultiSelect {
            border-radius: 5px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
        }
        .chat-bubble {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =================== Load ML Model =================== #
try:
    model = joblib.load("diabetes_model.pkl")
except FileNotFoundError:
    st.error("❌ Error: ML model file not found!")

# =================== Google Gemini API Key =================== #
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    if api_key:
        genai.configure(api_key=api_key)

# =================== Streamlit UI =================== #
st.title("🩺 Smart Healthcare Assistant")
st.write("🔹 Diabetes Prediction | 🏥 Personalized Healthcare | 🤖 AI Chatbot")

# Tabs
tab1, tab2, tab3 = st.tabs(["Diabetes Prediction", "Personalized Healthcare", "AI Chatbot"])

# =================== 🩸 TAB 1: DIABETES PREDICTION =================== #
with tab1:
    st.header("🔍 Diabetes Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=500, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, step=1)

    if st.button("Predict Diabetes", key="predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        try:
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.error("⚠️ High Risk of Diabetes Detected!")
            else:
                st.success("✅ No Diabetes Detected.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =================== 🏥 TAB 2: PERSONALIZED HEALTHCARE =================== #
with tab2:
    st.header("🏥 Personalized Healthcare Recommendations")
    lifestyle = st.selectbox("Select Lifestyle", ["Sedentary", "Moderately Active", "Highly Active"])
    symptoms = st.multiselect("Select Symptoms", ["Frequent Urination", "Excessive Thirst", "Fatigue", "Blurred Vision", "Slow Healing Wounds"])
    if st.button("Get Recommendations", key="recommend"):
        recs = []
        if lifestyle == "Sedentary":
            recs.append("Consider light exercises like walking or yoga.")
        elif lifestyle == "Moderately Active":
            recs.append("Maintain your current activity level but track your glucose levels.")
        elif lifestyle == "Highly Active":
            recs.append("Ensure proper hydration and balanced nutrition.")
        if "Frequent Urination" in symptoms:
            recs.append("Drink enough water and monitor blood sugar levels regularly.")
        if "Fatigue" in symptoms:
            recs.append("Consider adding more protein to your diet and check for deficiencies.")
        if recs:
            st.write("### 🏥 Your Recommendations:")
            for rec in recs:
                st.write(f"✔ {rec}")
        else:
            st.success("You seem to be in good health! Keep maintaining a healthy lifestyle.")

# =================== 🤖 TAB 3: AI CHATBOT =================== #
with tab3:
    st.header("🤖 AI Health Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.markdown(f"<div class='chat-bubble'><strong>{msg['role'].capitalize()}:</strong> {msg['content']}</div>", unsafe_allow_html=True)
    user_input = st.text_input("Ask your health question:")
    if st.button("Ask AI", key="ask"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            if api_key:
                try:
                    model = genai.GenerativeModel("gemini-1.5-pro-latest")  
                    response = model.generate_content(user_input)
                    ai_response = response.text if hasattr(response, 'text') else "⚠️ AI could not generate a response."
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.markdown(f"<div class='chat-bubble'><strong>AI:</strong> {ai_response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ AI Error: {e}")
            else:
                st.error("❌ Please enter a valid Gemini API Key!")
