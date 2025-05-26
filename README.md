FeelTrove – AI-Powered Sentiment & Emotion Analysis Dashboard
FeelTrove is an intelligent platform that empowers businesses and researchers to unlock deep emotional insights from text data. It uses cutting-edge natural language processing (NLP), sentiment analysis, and emotion detection models to visualize and understand feedback like never before.
feel trove logo
![logo](https://github.com/user-attachments/assets/e324834d-2ac1-43ee-86d6-d68722fe57c2)


🔍 Features
🔐 Secure login-based access to dashboard

📊 Real-time sentiment analytics with Plotly visualizations

🌍 Multilingual support using Google Translate API

😄 Emotion detection with HuggingFace Transformers

☁️ Upload CSV reviews for custom analysis

✍️ Write your own review to get instant feedback

🔁 Toggle between dark and light themes

📅 Time-based filtering and weekday analysis

☁️ Word cloud and keyword frequency visuals

Explore full feature visuals on the Feature Page.

🧰 Tech Stack
Frontend: HTML, Tailwind CSS, JavaScript (AOS, GSAP, Typed.js)

Backend: Python, Streamlit

AI Models: nltk.SentimentIntensityAnalyzer, j-hartmann/emotion-english-distilroberta-base (HuggingFace)

Libraries: pandas, numpy, plotly, wordcloud, matplotlib, nltk, transformers, deep-translator

🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/feeltrove.git
cd feeltrove
2. Install Dependencies
Make sure you’re using Python 3.10+.

bash
Copy
Edit
pip install -r requirements.txt
3. Run the App
bash
Copy
Edit
streamlit run app1.py
📂 Folder Structure
bash
Copy
Edit
.
├── app1.py                # Main Streamlit app
├── index.html             # Landing page (static)
├── features.html          # Feature showcase (static)
├── requirements.txt       # Required Python packages
├── images/                # Static assets (logo, graphs, illustrations)

✨ Demo Screenshots
![newplot (2)](https://github.com/user-attachments/assets/f50e005b-cb33-4a38-8b25-c06ee32e9d2d)
![newplot (3)](https://github.com/user-attachments/assets/e1cbabba-683f-4ddb-8e3b-7b943fe02760)

🔐 Login Credentials (for demo)
Username: admin

Password: 1234

Note: You can change credentials inside app1.py > login() function.

👨‍💻 Developer
Abubakker Siddique
Founder & Developer of FeelTrove
LinkedIn | Twitter
