#  SentiScope – AI Review Intelligence Platform

An AI-powered analytics platform that transforms raw customer reviews into actionable business insights using Natural Language Processing and Machine Learning.

## 🌐 Live Demo
👉 https://sentiscope19.streamlit.app/
---

## 🚀 Project Overview

Customer ratings often fail to reflect real user experience.
**SentiScope bridges this gap** by analyzing review text to uncover hidden customer dissatisfaction and product issues.

It enables businesses to move from:

> ⭐ “What is the rating?”
> to
> 🧠 “Why are customers unhappy?”

---

## 🎯 Key Features

* ✅ Sentiment Analysis using VADER (rule-based NLP)
* ✅ Machine Learning Model (TF-IDF + Logistic Regression)
* ✅ VADER vs ML Sentiment Comparison
* ✅ Aspect-Based Complaint Detection (battery, price, delivery, etc.)
* ✅ Rating vs Sentiment Mismatch Detection (fake/misleading reviews)
* ✅ AI-generated Insights & Recommendations
* ✅ Interactive Dashboard (Filters, Search, Sorting)
* ✅ Exportable filtered dataset

---

## 🧠 Core Idea

> High ratings do not always mean satisfied customers.

SentiScope detects **hidden dissatisfaction patterns** by comparing textual sentiment with numerical ratings.

---

## 📊 Dashboard Capabilities

* 🔍 Brand-level filtering
* 🔎 Real-time review search
* 📊 Sentiment distribution visualization
* 🚨 Top complaint identification
* 🤖 ML vs Rule-based model comparison
* ⚠️ Suspicious review detection
* 🏆 Top-performing products
* 🔥 Worst reviews explorer

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Data Processing:** Pandas
* **NLP:** VADER Sentiment Analysis
* **Machine Learning:** TF-IDF + Logistic Regression
* **Visualization:** Streamlit Charts

---

## 📁 Project Structure

```
SentiScope/
├── app/
│   └── app.py
|   └── final_reviews.csv
├── data/
│   ├── reviews.csv
│   ├── cleaned_reviews.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   └── .gitkeep
├── README.md
└── requirements.txt
```

---

## ⚠️ Key Insights Generated

* Many products show **high ratings but negative sentiment**
* Detection of **inconsistent or misleading reviews**
* ML model highlights **bias patterns in sentiment classification**
* Identification of major customer pain points:

  * 🔋 Battery
  * 💸 Price
  * 🚚 Delivery

---

## 📈 Business Impact

* Helps companies identify **true customer pain points**
* Improves **product decision-making**
* Detects **review manipulation or inconsistencies**
* Enables **data-driven product improvements**

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the app

```
streamlit run app/app.py
```

---

## 🚀 Future Improvements

* Integrate transformer-based models (BERT)
* Improve aspect extraction using NLP models
* Handle class imbalance in ML training
* Deploy as a live web application

---

## 👩‍💻 Author

Sneha

---

## 🏆 Conclusion

SentiScope is an end-to-end AI-powered analytics platform that goes beyond ratings to uncover the **true voice of the customer**, enabling smarter and more informed business decisions.
