import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Configure Gemini API key from Streamlit Cloud secrets
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="GATE CSE Topic Predictor (Streamlit Cloud)", layout="wide")

st.title("🚀 GATE CSE Topic Predictor with Year-wise Analysis & Study Roadmap")
st.write("The GATE dataset is loaded directly from GitHub.")

# GitHub raw CSV URL (update this with your actual repo link)
GITHUB_CSV_URL = "https://github.com/devanshvpurohit/Gatepredictionandanalysis-/blob/main/questions-data.csv"

@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce load
def load_data():
    df = pd.read_csv(GITHUB_CSV_URL)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data from GitHub: {e}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Check required columns
required_cols = {'question', 'topic', 'subject', 'year'}
if not required_cols.issubset(set(df.columns)):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

# Convert year column to int (if not)
df['year'] = df['year'].astype(int)

# Year-wise topic frequency for all subjects
st.subheader("📈 Year-wise Topic Frequency Analysis")

# Select subject to analyze
subject_list = df['subject'].unique().tolist()
selected_subject = st.selectbox("Select Subject to Analyze", subject_list)

subject_df = df[df['subject'] == selected_subject]

# Create pivot table: years vs topics counts
pivot = pd.pivot_table(subject_df, index='year', columns='topic', aggfunc='size', fill_value=0)

st.dataframe(pivot)

# Prepare prompt for Gemini
topic_trends_text = ""
for year in sorted(pivot.index):
    topics_count = pivot.loc[year]
    topics_str = ", ".join([f"{topic}({count})" for topic, count in topics_count.items() if count > 0])
    topic_trends_text += f"{year}: {topics_str}\n"

prompt = f"""
You are a GATE CSE subject expert analyzing historical trends for the subject '{selected_subject}'.
Here are the topic frequencies by year:

{topic_trends_text}

Based on these trends, predict the top 5 topics most likely to appear in the next exam for {selected_subject}.

Also, provide a concise 4-7 months study roadmap to maximize marks in {selected_subject}, mentioning key topics and study strategies.
"""

if st.button(f"Predict Topics & Generate Roadmap for {selected_subject}"):
    with st.spinner("Generating predictions and roadmap via Gemini..."):
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            st.subheader(f"🎯 Gemini Predictions & Study Roadmap for {selected_subject}")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error with Gemini API: {e}")

# Optional: show overall topic frequency across all subjects
st.subheader("🔎 Overall Topic Frequency Across All Subjects")
overall_pivot = pd.pivot_table(df, index='year', columns='topic', aggfunc='size', fill_value=0)
st.dataframe(overall_pivot)
