# Updated Feel Trove Sentiment Analysis Dashboard with Enhanced Interactive Visuals

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import re
from deep_translator import GoogleTranslator
from transformers import pipeline
import seaborn as sns
import sqlalchemy

# Downloads
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initializations
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
emotion_model = pipeline(
    "text-classification", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    return_all_scores=False,
    device=-1
)

# Page config
st.set_page_config(page_title="Feel Trove Dashboard", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.title("🔐 Login to Feel Trove")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "feel trove":
            st.session_state["authenticated"] = True
            st.success("Login successful! Redirecting to dashboard...")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- Logout Button ---
if st.sidebar.button("🚪 Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

# Theme
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

def set_theme():
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

st.sidebar.button("Toggle Theme", on_click=set_theme)
BG_COLOR = "#111" if st.session_state["theme"] == 'dark' else "#fff"
TEXT_COLOR = "#fff" if st.session_state["theme"] == 'dark' else "#000"

# --- SQL Load Function ---
def load_sql_data(db_type, host, port, user, password, db_name, table_name):
    try:
        if db_type == "MySQL":
            engine_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
        elif db_type == "PostgreSQL":
            engine_str = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        elif db_type == "SQLite":
            engine_str = f"sqlite:///{db_name}"
        else:
            st.error("Unsupported database type.")
            return None
        engine = sqlalchemy.create_engine(engine_str)
        df = pd.read_sql_table(table_name, engine)
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# --- SQL Insert Function ---
def insert_review(name, product, review, sentiment, emotion, conn_info):
    try:
        if conn_info["db_type"] == "MySQL":
            engine_str = f"mysql+mysqlconnector://{conn_info['user']}:{conn_info['password']}@{conn_info['host']}:{conn_info['port']}/{conn_info['db_name']}"
        else:
            st.error("Only MySQL insert is currently supported.")
            return False
        engine = sqlalchemy.create_engine(engine_str)
        now = datetime.now()
        df_insert = pd.DataFrame([{
            "Name": name,
            "Product": product,
            "Review": review,
            "Date": now,
            "Sentiment": sentiment,
            "Emotion": emotion,
            "Score": 1 if sentiment == "Positive" else -1 if sentiment == "Negative" else 0
        }])
        df_insert.to_sql(conn_info['table_name'], con=engine, if_exists='append', index=False)
        return True
    except Exception as e:
        st.error(f"Insert failed: {e}")
        return False

# Generate synthetic reviews
@st.cache_data
def generate_reviews(n=1000):
    np.random.seed(42)
    products = ['Phone', 'Laptop', 'Headphones', 'Camera', 'Smartwatch']
    sentiments = ['Positive', 'Neutral', 'Negative']
    templates = {
        'Positive': ['Love it!', 'Great product', 'Highly recommend', 'Excellent!', 'Very satisfied'],
        'Neutral': ['It’s okay', 'Average', 'Not bad', 'Neutral', 'Fair product'],
        'Negative': ['Horrible', 'Very disappointed', 'Not worth it', 'Terrible', 'Wouldn’t buy again']
    }
    data = []
    for _ in range(n):
        sent = np.random.choice(sentiments, p=[0.5, 0.2, 0.3])
        review = np.random.choice(templates[sent])
        score = {'Positive': 1, 'Neutral': 0, 'Negative': -1}[sent]
        data.append({
            'Name': f'User{np.random.randint(100, 999)}',
            'Product': np.random.choice(products),
            'Review': review,
            'Date': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2024-12-31'))),
            'Sentiment': sent,
            'Score': score
        })
    return pd.DataFrame(data)

def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    return Counter(keywords).most_common(10)

def detect_sentiment_multilingual(review):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(review)
        score = sia.polarity_scores(translated)['compound']
        return "Positive" if score > 0.3 else "Negative" if score < -0.3 else "Neutral"
    except:
        return "Unknown"

def get_emotion(review):
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(review)
        result = emotion_model(translated)
        return result[0]['label'] if result else "Unknown"
    except:
        return "Unknown"

# Load synthetic data
df = generate_reviews()
df["Weekday"] = df["Date"].dt.day_name()

# SQL Import Panel
with st.sidebar.expander("Connect to SQL Database"):
    db_type = st.selectbox("Database Type", ["SQLite", "MySQL", "PostgreSQL"])
    host = st.text_input("Host", "localhost")
    port = st.text_input("Port", "3306" if db_type == "MySQL" else "5432")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    db_name = st.text_input("Database Name")
    table_name = st.text_input("Table Name (e.g., reviews)")
    if st.button("Load SQL Data"):
        sql_df = load_sql_data(db_type, host, port, user, password, db_name, table_name)
        if sql_df is not None and "Review" in sql_df.columns and "Product" in sql_df.columns:
            st.session_state["sql_connection"] = {
                "db_type": db_type, "host": host, "port": port, "user": user,
                "password": password, "db_name": db_name, "table_name": table_name
            }
            sql_df["Sentiment"] = sql_df["Review"].apply(detect_sentiment_multilingual)
            sql_df["Emotion"] = sql_df["Review"].apply(get_emotion)
            sql_df["Date"] = pd.to_datetime(sql_df.get("Date", pd.Timestamp.today()))
            sql_df["Score"] = sql_df["Sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
            st.dataframe(sql_df)

# Navigation
page = st.sidebar.radio("Navigate", ["Dashboard", "Upload CSV", "Write a Review"])


# Move this block inside the `if page == "Dashboard":` section to restrict filtering to dashboard only

if page == "Dashboard":
    st.title("📊 Feel Trove Sentiment Analysis Dashboard")

    # Filter
    st.sidebar.title("Filter Data")
    products = st.sidebar.multiselect("Select Product", df["Product"].unique(), default=list(df["Product"].unique()))
    sentiments = st.sidebar.multiselect("Select Sentiment", df["Sentiment"].unique(), default=list(df["Sentiment"].unique()))
    date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])

    filtered_df = df[
        (df["Product"].isin(products)) &
        (df["Sentiment"].isin(sentiments)) &
        (df["Date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(filtered_df))
    col2.metric("Positive", (filtered_df["Sentiment"] == "Positive").sum())
    col3.metric("Negative", (filtered_df["Sentiment"] == "Negative").sum())

    st.subheader("1. Sentiment Distribution")
    st.plotly_chart(px.pie(filtered_df, names="Sentiment", title="Sentiment Distribution"), use_container_width=True)

    st.subheader("2. Reviews by Product")
    st.plotly_chart(px.histogram(filtered_df, x="Product", color="Sentiment", barmode="group", title="Reviews by Product"), use_container_width=True)

    st.subheader("3. Sentiment Over Time")
    st.plotly_chart(px.line(filtered_df.groupby("Date")["Score"].mean().reset_index(), x="Date", y="Score", title="Average Sentiment Over Time"), use_container_width=True)

    st.subheader("4. Sentiment by Weekday")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = filtered_df["Weekday"].value_counts().reindex(weekday_order, fill_value=0).reset_index()
    weekday_counts.columns = ["Weekday", "Count"]
    st.plotly_chart(px.bar(weekday_counts, x="Weekday", y="Count", title="Reviews by Day of Week"), use_container_width=True)

    st.subheader("5. Word Cloud")
    text = " ".join(filtered_df["Review"])
    wordcloud = WordCloud(width=800, height=400, background_color='black' if st.session_state["theme"] == 'dark' else 'white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("6. Keyword Frequency")
    st.dataframe(pd.DataFrame(extract_keywords(text), columns=["Keyword", "Frequency"]))

    st.subheader("7. Sentiment Composition by Product")
    st.plotly_chart(px.histogram(filtered_df, x="Product", color="Sentiment", barmode="stack", title="Sentiment Composition by Product"), use_container_width=True)

    st.subheader("8. Sentiment Score Distribution per Product")
    st.plotly_chart(px.box(filtered_df, x="Product", y="Score", color="Product", title="Sentiment Score Distribution per Product"), use_container_width=True)

elif page == "Upload CSV":
    st.title("Upload Your CSV File")
    uploaded = st.file_uploader("Upload CSV with 'Review' and 'Product' columns", type='csv')
    if uploaded:
        user_df = pd.read_csv(uploaded)
        if 'Review' in user_df.columns and 'Product' in user_df.columns:
            user_df["Sentiment"] = user_df["Review"].apply(detect_sentiment_multilingual)
            user_df["Emotion"] = user_df["Review"].apply(get_emotion)
            user_df["Date"] = pd.to_datetime(user_df.get("Date", pd.Timestamp.today()))
            user_df["Score"] = user_df["Sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
            user_df["Weekday"] = user_df["Date"].dt.day_name()
            st.success("Analysis complete!")
            st.dataframe(user_df)

            # ✅ Now all 9 visualizations properly indented:
            st.subheader("1. Sentiment Distribution")
            st.plotly_chart(px.pie(user_df, names="Sentiment", title="Sentiment Distribution"), use_container_width=True)

            st.subheader("2. Reviews by Product")
            st.plotly_chart(px.histogram(user_df, x="Product", color="Sentiment", barmode="group", title="Reviews by Product"), use_container_width=True)

            st.subheader("3. Emotion Distribution")
            st.plotly_chart(px.histogram(user_df, x="Emotion", title="Emotion Distribution"), use_container_width=True)

            st.subheader("4. Reviews by Day of Week")
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = user_df["Weekday"].value_counts().reindex(weekday_order, fill_value=0).reset_index()
            weekday_counts.columns = ["Weekday", "Count"]
            st.plotly_chart(px.bar(weekday_counts, x="Weekday", y="Count", title="Reviews by Day of Week"), use_container_width=True)

            st.subheader("5. Sentiment Volume Over Time")
            volume = user_df.groupby(["Date", "Sentiment"]).size().reset_index(name="Count")
            st.plotly_chart(px.area(volume, x="Date", y="Count", color="Sentiment", title="Sentiment Volume Over Time"), use_container_width=True)

            st.subheader("6. Sentiment Composition by Product")
            st.plotly_chart(px.histogram(user_df, x="Product", color="Sentiment", barmode="stack", title="Sentiment Composition by Product"), use_container_width=True)

            st.subheader("7. Sentiment Score Distribution per Product")
            st.plotly_chart(px.box(user_df, x="Product", y="Score", color="Product", title="Sentiment Score Distribution per Product"), use_container_width=True)

            st.subheader("8. Word Cloud from Reviews")
            text = " ".join(user_df["Review"])
            wordcloud = WordCloud(width=800, height=400, background_color='black' if st.session_state["theme"] == 'dark' else 'white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            st.subheader("9. Keyword Frequency")
            st.write(pd.DataFrame(extract_keywords(text), columns=["Keyword", "Frequency"]))
        else:
            st.error("Missing required columns: 'Review' and 'Product'")
           

elif page == "Write a Review":
    st.title("Submit a Review")
    name = st.text_input("Your Name")
    product = st.selectbox("Product", df["Product"].unique())
    review = st.text_area("Write your review")

    if st.button("Analyze"):
        if review:
            sentiment = detect_sentiment_multilingual(review)
            emotion = get_emotion(review)
            st.success(f"Sentiment: {sentiment} | Emotion: {emotion}")
            if "sql_connection" in st.session_state:
                if insert_review(name, product, review, sentiment, emotion, st.session_state["sql_connection"]):
                    st.success("Review saved to database.")
                else:
                    st.error("Failed to save review.")
            else:
                st.info("Connect to SQL database first to enable saving.")
        else:
            st.warning("Please write a review first.")
