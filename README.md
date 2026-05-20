# 📈 Live Hacker News Trend & Sentiment Analyzer

A modern, real-time natural language processing (NLP) dashboard that fetches top stories from the Hacker News API, evaluates headline sentiment scores using NLTK VADER, and visualizes upvote distributions, sentiment breakdowns, and trending keywords.

🔗 **Live Demo:** [live-api-trend-sentiment-analyzer.streamlit.app](https://live-api-trend-sentiment-analyzer.streamlit.app/)

---

## ✨ Features

- **🚀 Real-Time Pipeline**: Automatically fetches live data directly from the official Hacker News Firebase API.
- **🧠 NLP Sentiment Engine**: Processes article headlines using NLTK's `SentimentIntensityAnalyzer` (VADER) to score and categorize sentiment as *Positive*, *Neutral*, or *Negative*.
- **💎 Premium UI/UX Dashboard**: A beautiful, responsive glassmorphism dark-themed dashboard built with **Streamlit** and custom CSS.
- **🎨 Interactive Visualizations**:
  - **Headline Sentiment vs. Upvotes**: Plotly scatter plot mapping sentiment score (-1.0 to 1.0) against article upvotes with custom hover tooltips.
  - **Sentiment Breakdown**: Donut chart illustrating the percentage distribution of positive, negative, and neutral headlines.
  - **Trending Keywords**: Horizontal bar chart identifying popular topics/words in the headlines (filtered for common stop-words).
- **🔍 Granular Filters & Search**:
  - Filter stories by minimum upvotes or sentiment categories in real-time.
  - Live fuzzy search box to instantly search through article headlines.
  - Complete, sortable data table for detailed inspection.
- **🖥️ Standalone CLI Script**: A lightweight script (`hn_trends.py`) that performs the analysis and opens a local HTML browser window with Plotly charts.

---

## 🛠️ Technology Stack

- **Core**: Python 3.13+
- **Front-End Dashboard**: Streamlit
- **Data Manipulation**: Pandas
- **Natural Language Processing**: NLTK (VADER Lexicon)
- **Data Visualization**: Plotly Express
- **Networking**: Requests

---

## 📂 Project Structure

```text
├── app.py             # Main Streamlit web application with custom CSS & components
├── hn_trends.py       # Standalone CLI Python script for quick visualization
├── requirements.txt   # Pip package dependencies list
└── README.md          # Project documentation (this file)
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/-Live-API-Trend-Sentiment-Analyzer.git
cd -Live-API-Trend-Sentiment-Analyzer
```

### 2. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Launch the Web Dashboard (Recommended)
To run the interactive Streamlit dashboard:
```bash
python -m streamlit run app.py
```
This will start the local web server. Open your browser and navigate to:
```text
http://localhost:8501
```

### Option B: Run Standalone CLI Script
To quickly test the scraper and view the charts without starting a Streamlit server:
```bash
python hn_trends.py
```
This script runs in your console and will automatically open a local HTML chart in your default web browser.

---

## ⚙️ Troubleshooting

### NumPy 2.x / Version Mismatch Issue
If you run into import errors indicating that modules (like `xarray`, `numexpr`, or `bottleneck`) compiled using NumPy 1.x cannot run with a NumPy 2.x installation, follow these options:

1. **Use System Python**: Ensure you are running the app using your system-wide Python environment (which has updated packages):
   ```bash
   python -m streamlit run app.py
   ```
2. **Upgrade Packages in Environment**: If using an environment like Anaconda, update the dependencies to compile with NumPy 2.x support:
   ```bash
   pip install --upgrade pandas xarray plotly numexpr bottleneck
   ```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
