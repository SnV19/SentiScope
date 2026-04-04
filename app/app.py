import streamlit as st
import pandas as pd
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ast
import os

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Review Intelligence Platform", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "final_reviews.csv")
    return pd.read_csv(file_path)

df = load_data()

# ---------------- VADER ----------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_vader(text):
    return analyzer.polarity_scores(str(text))['compound']

def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_score'] = df['clean_review'].apply(get_sentiment_vader)
df['sentiment'] = df['sentiment_score'].apply(label_sentiment)

# ---------------- ML MODEL ----------------
@st.cache_resource
def train_model(df):
    df = df.dropna(subset=["clean_review", "sentiment"])

    X = df["clean_review"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model(df)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("SentiScope")
st.markdown("###  AI-powered customer insight engine for product analysis")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

brand_filter = st.sidebar.selectbox("Select Brand", df['brand'].dropna().unique())
filtered_df = df[df['brand'] == brand_filter].copy()

# Search
search = st.sidebar.text_input("Search reviews")
if search:
    filtered_df = filtered_df[
        filtered_df['clean_review'].str.contains(search, case=False, na=False)
    ]

# Sort
sort_option = st.sidebar.selectbox("Sort by", ["None", "Rating", "Sentiment"])

if sort_option == "Rating":
    filtered_df = filtered_df.sort_values(by="rating", ascending=False)

elif sort_option == "Sentiment":
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    filtered_df["sentiment_num"] = filtered_df["sentiment"].map(sentiment_map)
    filtered_df = filtered_df.sort_values(by="sentiment_num", ascending=False)

filtered_df.drop(columns=['sentiment_num'], inplace=True, errors='ignore')

# ---------------- EMPTY CHECK ----------------
if len(filtered_df) == 0:
    st.warning("No data available for selected filters.")
    st.stop()

# ---------------- ML PREDICTION ----------------
X_filtered = vectorizer.transform(filtered_df["clean_review"])
filtered_df["ml_sentiment"] = model.predict(X_filtered)

# ---------------- ADVANCED MISMATCH DETECTION ----------------
def detect_mismatch(row):
    rating = row['rating']
    sentiment = row['sentiment']

    if rating <= 2 and sentiment == "Positive":
        return "Highly Suspicious"
    elif rating >= 4 and sentiment == "Negative":
        return "Highly Suspicious"
    elif rating == 3 and sentiment in ["Positive", "Negative"]:
        return "Moderate"
    elif rating >= 4 and sentiment == "Neutral":
        return "Slight"
    elif rating <= 2 and sentiment == "Neutral":
        return "Slight"
    else:
        return "Normal"

filtered_df['mismatch_level'] = filtered_df.apply(detect_mismatch, axis=1)

# ---------------- SUSPICION SCORE ----------------
def suspicion_score(row):
    score = 0

    if row['mismatch_level'] == "Highly Suspicious":
        score += 2
    elif row['mismatch_level'] == "Moderate":
        score += 1

    if len(str(row['clean_review']).split()) < 5:
        score += 1

    return score

filtered_df['suspicion_score'] = filtered_df.apply(suspicion_score, axis=1)

# ---------------- KPIs ----------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("📦 Total Reviews", len(filtered_df))
col2.metric("⭐ Avg Rating", round(filtered_df['rating'].mean(), 2))
col3.metric("😡 Negative %", round((filtered_df['sentiment']=="Negative").mean()*100, 2))

st.divider()

# ---------------- SENTIMENT + COMPLAINTS ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("😊 Sentiment Distribution")
    st.bar_chart(filtered_df['sentiment'].value_counts())

# ---------------- TOP COMPLAINTS ----------------
all_aspects = []

for row in filtered_df[filtered_df['sentiment'] == "Negative"]['aspects']:
    if isinstance(row, str):
        try:
            row = ast.literal_eval(row)
        except:
            row = []
    all_aspects.extend(row)

aspect_counts = Counter(all_aspects)

with col2:
    st.subheader("🚨 Top Complaints")
    if aspect_counts:
        st.bar_chart(dict(aspect_counts))
    else:
        st.write("No major complaints found.")

st.divider()

# ---------------- ML VS VADER ----------------
st.subheader("🤖 VADER vs ML Comparison")
comparison = pd.crosstab(filtered_df['sentiment'], filtered_df['ml_sentiment'])
st.write(comparison)

# ---------------- AI INSIGHTS ----------------
# ---------------- SMART AI INSIGHTS ----------------
st.subheader("🧠 AI Insights (Advanced)")

avg_rating = filtered_df['rating'].mean()
negative_ratio = (filtered_df['sentiment'] == "Negative").mean()
positive_ratio = (filtered_df['sentiment'] == "Positive").mean()
high_suspicion_ratio = (filtered_df['suspicion_score'] >= 2).mean()

# ---------------- OVERALL HEALTH ----------------
if avg_rating >= 4 and negative_ratio < 0.2:
    st.success("🟢 Strong product performance with high customer satisfaction and low complaints.")
elif avg_rating >= 3:
    st.info("🟡 Product shows mixed feedback — improvements needed in certain areas.")
else:
    st.error("🔴 Product performance is poor with significant dissatisfaction among customers.")

# ---------------- SENTIMENT vs RATING GAP ----------------
if avg_rating >= 4 and negative_ratio > 0.3:
    st.warning("⚠️ High ratings but many negative reviews — possible rating inflation or misleading feedback.")
elif avg_rating <= 2.5 and positive_ratio > 0.3:
    st.warning("⚠️ Low ratings but many positive reviews — inconsistent customer perception detected.")

# ---------------- TOP ISSUE ----------------
if aspect_counts:
    top_issue, count = aspect_counts.most_common(1)[0]
    issue_percent = round((count / len(filtered_df)) * 100, 2)

    st.write(f"🚨 Major issue impacting customers: **{top_issue}** ({issue_percent}% of reviews)")

    # Actionable suggestions
    if top_issue.lower() == "battery":
        st.write("🔋 Improving battery performance can significantly boost customer satisfaction.")
    elif top_issue.lower() == "price":
        st.write("💸 Customers perceive pricing as high — consider discounts or value justification.")
    elif top_issue.lower() == "delivery":
        st.write("🚚 Delivery delays/issues are hurting user experience — optimize logistics.")
    elif top_issue.lower() == "quality":
        st.write("⚙️ Product quality concerns detected — focus on durability and consistency.")

# ---------------- FAKE REVIEW SIGNAL ----------------
if high_suspicion_ratio > 0.3:
    st.error("🚨 High probability of fake or manipulated reviews detected. Data reliability is questionable.")
elif high_suspicion_ratio > 0.15:
    st.warning("⚠️ Some suspicious review patterns detected — interpret insights carefully.")
else:
    st.success("✅ Review data appears reliable with minimal suspicious activity.")

# ---------------- TREND INSIGHT ----------------
sentiment_trend = filtered_df['sentiment'].value_counts()

if "Negative" in sentiment_trend and "Positive" in sentiment_trend:
    if sentiment_trend["Negative"] > sentiment_trend["Positive"]:
        st.warning("📉 Negative sentiment dominates — urgent improvements required.")
    else:
        st.success("📈 Positive sentiment dominates — product is well received overall.")

# ---------------- PRODUCT LEVEL ALERT ----------------
problem_products = (
    filtered_df[filtered_df['sentiment']=="Negative"]
    ['product']
    .value_counts()
)

if len(problem_products) > 0:
    worst_product = problem_products.idxmax()
    st.write(f"🚨 Most problematic product: **{worst_product}** (highest negative feedback)")

# ---------------- BUSINESS SUMMARY ----------------
st.markdown("### 📌 Executive Summary")

# Top product (best rated)
top_product = (
    filtered_df.groupby('product')['rating']
    .mean()
    .idxmax()
)

# Worst product (most negative reviews)
problem_products = (
    filtered_df[filtered_df['sentiment']=="Negative"]
    ['product']
    .value_counts()
)

worst_product = problem_products.idxmax() if len(problem_products) > 0 else None

# Summary logic
if avg_rating >= 4 and high_suspicion_ratio < 0.1:
    st.write(f"**{top_product}** is performing strongly with genuine positive customer feedback.")
elif avg_rating >= 3:
    st.write(f"Products like **{worst_product}** need targeted improvements to reduce negative experiences.")
else:
    st.write(f"Products such as **{worst_product}** are underperforming and need immediate attention.")
# ---------------- FAKE REVIEW DETECTION DASHBOARD ----------------
st.divider()
st.subheader("🚨 Fake / Misleading Review Detection")

col1, col2 = st.columns(2)

with col1:
    st.write("### Mismatch Levels")
    st.bar_chart(filtered_df['mismatch_level'].value_counts())

with col2:
    st.write("### Suspicion Score Distribution")
    st.bar_chart(filtered_df['suspicion_score'].value_counts())

# ---------------- RELIABILITY MESSAGE ----------------
high_suspicion_ratio = (filtered_df['suspicion_score'] >= 2).mean()

if high_suspicion_ratio > 0.25:
    st.error("🚨 High number of potentially fake or misleading reviews detected.")
elif high_suspicion_ratio > 0.1:
    st.warning("⚠️ Some suspicious review patterns detected.")
else:
    st.success("✅ Reviews appear mostly reliable.")

# ---------------- SUSPICIOUS REVIEWS ----------------
st.subheader("🕵️ Suspicious Reviews")

suspicious_reviews = filtered_df[
    filtered_df['suspicion_score'] >= 2
].sort_values(by='suspicion_score', ascending=False)

num_suspicious = st.slider("Number of suspicious reviews to show", 5, 20, 10)

for _, row in suspicious_reviews.head(num_suspicious).iterrows():
    st.markdown(f"""
    **📦 Product:** {row['product']}  
    ⭐ Rating: {row['rating']} | 😡 Sentiment: {row['sentiment']}  
    ⚠️ Mismatch: {row['mismatch_level']} | 🔍 Score: {row['suspicion_score']}  
    📝 Review: {row['clean_review']}
    """)
    st.markdown("---")

st.divider()

# ---------------- DOWNLOAD ----------------
st.download_button(
    label="📥 Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_reviews.csv"
)

# ---------------- TOP PRODUCTS ----------------
st.subheader("🏆 Top Products")

top_products = (
    filtered_df.groupby('product')['rating']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

st.bar_chart(top_products)

# ---------------- PROBLEMATIC PRODUCTS ----------------
st.subheader("🚨 Most Problematic Products")

problem_products = (
    filtered_df[filtered_df['sentiment']=="Negative"]
    ['product']
    .value_counts()
    .head(10)
)

st.bar_chart(problem_products)

# ---------------- WORST REVIEWS ----------------
st.subheader("🔥 Worst Reviews Explorer")

num_reviews = st.slider("Select number of worst reviews", 5, 20, 10)

worst_reviews = filtered_df[filtered_df['sentiment'] == "Negative"]
worst_reviews = worst_reviews.sort_values(by="sentiment_score").head(num_reviews)

for _, row in worst_reviews.iterrows():
    st.markdown(f"""
    **📦 Product:** {row['product']}  
    ⭐ Rating: {row['rating']} | 😡 Sentiment: {row['sentiment']}  
    📝 Review: {row['clean_review']}
    """)
    st.markdown("---")