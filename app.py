import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
le_bacteria = pickle.load(open("bacteria_encoder.pkl", "rb"))
le_antibiotic = pickle.load(open("antibiotic_encoder.pkl", "rb"))

# Page config
st.set_page_config(page_title="Antibiotic AI", layout="centered")

# 🌙 DARK MODE STYLE
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# 🎨 HEADER
st.markdown("<h1 style='text-align: center; color: #00FFAA;'>🧬 Antibiotic Resistance AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered prediction & treatment recommendation</p>", unsafe_allow_html=True)

st.divider()

# 📥 INPUT
col1, col2 = st.columns(2)

with col1:
    bacteria = st.selectbox("🦠 Select Bacteria", le_bacteria.classes_)

with col2:
    antibiotic = st.selectbox("💊 Select Antibiotic", le_antibiotic.classes_)

st.divider()

# 🔍 PREDICTION
if st.button("🔍 Analyze"):

    b = le_bacteria.transform([bacteria])[0]
    a = le_antibiotic.transform([antibiotic])[0]

    pred = model.predict([[b, a]])
    prob = model.predict_proba([[b, a]])[0]

    st.subheader("🧪 Prediction Result")

    if pred[0] == 1:
        st.error("❌ Resistant")
        confidence = prob[1]
    else:
        st.success("✅ Susceptible")
        confidence = prob[0]

    st.write(f"**Confidence:** {round(confidence*100,2)}%")

    if confidence > 0.8:
        st.warning("⚠️ High Confidence")
    else:
        st.info("ℹ️ Moderate Confidence")

    st.divider()

    # 💡 Suggestions
    if pred[0] == 1:
        st.subheader("💡 Recommended Antibiotics")

        suggestions = []

        for drug in le_antibiotic.classes_:
            d = le_antibiotic.transform([drug])[0]
            p = model.predict([[b, d]])

            if p[0] == 0:
                suggestions.append(drug)

        if suggestions:
            st.success(", ".join(suggestions))
        else:
            st.warning("No effective antibiotic found")

    # 🤖 Explanation
    st.subheader("🤖 AI Explanation")
    if pred[0] == 1:
        st.write("The bacteria shows resistance patterns to similar antibiotics in training data.")
    else:
        st.write("The antibiotic is likely effective based on learned susceptibility patterns.")

# 📊 FEATURE IMPORTANCE
if st.checkbox("📊 Show Model Insights"):
    importance = model.feature_importances_
    features = ['Bacteria', 'Antibiotic']

    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    st.bar_chart(df_imp.set_index('Feature'))

# 🌍 HEATMAP
if st.checkbox("🌍 Show Resistance Heatmap"):

    bacteria_list = le_bacteria.classes_
    antibiotic_list = le_antibiotic.classes_

    data = []

    for b in bacteria_list:
        row = []
        b_enc = le_bacteria.transform([b])[0]

        for a in antibiotic_list:
            a_enc = le_antibiotic.transform([a])[0]
            pred = model.predict([[b_enc, a_enc]])[0]
            row.append(pred)

        data.append(row)

    df_heat = pd.DataFrame(data, index=bacteria_list, columns=antibiotic_list)

    fig, ax = plt.subplots()
    cax = ax.matshow(df_heat)

    plt.xticks(range(len(antibiotic_list)), antibiotic_list, rotation=90)
    plt.yticks(range(len(bacteria_list)), bacteria_list)

    plt.colorbar(cax)

    st.pyplot(fig)

# 📁 UPLOAD PATIENT DATA
st.divider()
st.subheader("📁 Upload Patient Data")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    st.write("Preview:")
    st.dataframe(df_upload.head())

    st.success("File uploaded successfully!")

# 📌 FOOTER
st.divider()
st.markdown("<p style='text-align:center;'>Built for smarter healthcare using AI</p>", unsafe_allow_html=True)