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

# Sort ------------------------------------------
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
st.subheader("🧠 AI Insights")

avg_rating = filtered_df['rating'].mean()

if avg_rating >= 4:
    st.success("Overall customer satisfaction is high.")
elif avg_rating >= 3:
    st.info("Customer feedback is mixed.")
else:
    st.warning("Customers are generally unhappy with this brand.")

# Top issue
if aspect_counts:
    top_issue = aspect_counts.most_common(1)[0][0]
    st.write(f"🚨 Most common issue: **{top_issue}**")

# Negative ratio
negative_ratio = (filtered_df['sentiment']=="Negative").mean()

if negative_ratio > 0.5:
    st.warning("More than 50% reviews are negative.")
elif negative_ratio > 0.3:
    st.info("Significant negative feedback detected.")

# ---------------- MISMATCH ----------------
mismatch = filtered_df[
    ((filtered_df['rating'] <= 2) & (filtered_df['sentiment'] == "Positive")) |
    ((filtered_df['rating'] >= 4) & (filtered_df['sentiment'] == "Negative"))
]

mismatch_ratio = len(mismatch) / len(filtered_df)

if mismatch_ratio > 0.2:
    st.warning("High mismatch between rating and sentiment — possible unreliable reviews.")

# Suggestions
if aspect_counts:
    if "battery" in aspect_counts:
        st.write("🔋 Improve battery performance to boost ratings.")
    if "price" in aspect_counts:
        st.write("💸 Pricing may be a concern for customers.")
    if "delivery" in aspect_counts:
        st.write("🚚 Delivery experience needs improvement.")

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