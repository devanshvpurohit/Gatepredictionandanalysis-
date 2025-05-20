import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# ✅ Configure Gemini API key from environment variable or Streamlit secrets
genai.configure(api_key=os.getenv("AIzaSyAUpmC2Vj5Zy_IygSEHLoZ8V3zDSPbgJRg"))

# ✅ Streamlit Page Config
st.set_page_config(page_title="GATE CSE Topic Predictor", layout="wide")
st.title("🚀 GATE CSE Topic Predictor with Year-wise Analysis & Study Roadmap")
st.write("The GATE dataset is loaded directly from GitHub (raw CSV format).")

# ✅ Correct Raw GitHub CSV URL
GITHUB_CSV_URL = "https://raw.githubusercontent.com/devanshvpurohit/Gatepredictionandanalysis-/main/questions-data.csv"

# ✅ Load data from GitHub (cached for 1 hour)
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(GITHUB_CSV_URL)
    return df

# ✅ Try loading the data
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Failed to load data from GitHub: {e}")
    st.stop()

# ✅ Preview the dataset
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ✅ Check required columns
required_cols = {'question', 'topic', 'subject', 'year'}
if not required_cols.issubset(df.columns):
    st.error(f"❌ CSV must contain columns: {required_cols}")
    st.stop()

# ✅ Ensure 'year' column is integer
df['year'] = df['year'].astype(int)

# ✅ Year-wise Topic Frequency Analysis
st.subheader("📈 Year-wise Topic Frequency Analysis")

# Select subject
subject_list = sorted(df['subject'].dropna().unique())
selected_subject = st.selectbox("📘 Select Subject", subject_list)

# Filter data for selected subject
subject_df = df[df['subject'] == selected_subject]

# Create pivot table: year vs topic count
pivot = pd.pivot_table(subject_df, index='year', columns='topic', aggfunc='size', fill_value=0)
st.dataframe(pivot)

# ✅ Prepare topic trends text for Gemini
topic_trends_text = ""
for year in sorted(pivot.index):
    row = pivot.loc[year]
    trends = ", ".join([f"{topic}({count})" for topic, count in row.items() if count > 0])
    topic_trends_text += f"{year}: {trends}\n"

# ✅ Gemini Prompt
prompt = f"""
You are a GATE CSE subject expert analyzing historical trends for the subject '{selected_subject}'.
Here are the topic frequencies by year:

{topic_trends_text}

Based on these trends, predict the top 5 topics most likely to appear in the next exam for {selected_subject}.

Also, provide a concise 4-7 months study roadmap to maximize marks in {selected_subject}, mentioning key topics and study strategies.
"""

# ✅ Generate Predictions and Roadmap
if st.button(f"🎯 Predict Topics & Generate Roadmap for {selected_subject}"):
    with st.spinner("🧠 Generating predictions using Gemini..."):
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            st.subheader(f"📌 Predictions & Study Plan for {selected_subject}")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Error with Gemini API: {e}")

# ✅ Optional: Overall Topic Frequency (All Subjects)
st.subheader("📊 Overall Topic Frequency Across All Subjects")
overall_pivot = pd.pivot_table(df, index='year', columns='topic', aggfunc='size', fill_value=0)
st.dataframe(overall_pivot)
