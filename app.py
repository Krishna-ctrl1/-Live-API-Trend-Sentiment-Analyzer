import streamlit as st
import requests
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import re
from collections import Counter

# Set page configurations
st.set_page_config(
    page_title="Hacker News Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for modern dark glassmorphism look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Font family overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Top Banner Header */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 2.2rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        text-align: center;
    }
    .header-title {
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Premium KPI Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.2rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        margin-bottom: 1.5rem;
        
        /* Fixed height and flexbox for perfect vertical and horizontal alignment */
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 12px 40px 0 rgba(96, 165, 250, 0.15);
    }
    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.2rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 600;
    }

    /* Headline Explorer Cards */
    .story-card {
        background: rgba(30, 41, 59, 0.35);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease-in-out;
    }
    .story-card:hover {
        background: rgba(30, 41, 59, 0.6);
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateX(4px);
    }
    .story-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .story-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f1f5f9;
        text-decoration: none;
        line-height: 1.4;
        transition: color 0.2s ease;
    }
    .story-title:hover {
        color: #60a5fa !important;
    }
    .story-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Sentiment & Score Badges */
    .badge {
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-neutral {
        background-color: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-score {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📈 Hacker News Sentiment Dashboard</h1>
    </div>
""", unsafe_allow_html=True)


# Data Fetcher
@st.cache_data(ttl=300)
def fetch_and_analyze(limit=50):
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    try:
        top_story_ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10).json()
    except Exception as e:
        st.error(f"Error fetching data from Hacker News API: {e}")
        return pd.DataFrame()
        
    stories_data = []
    progress_text = f"Pipelining live data from Hacker News ({limit} stories)..."
    progress_bar = st.progress(0, text=progress_text)
    
    for i, story_id in enumerate(top_story_ids[:limit]):
        try:
            story = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=5).json()
            if story and 'title' in story:
                stories_data.append({
                    'title': story.get('title'),
                    'score': story.get('score', 0),
                    'url': story.get('url', ''),
                    'timestamp': story.get('time')
                })
        except Exception:
            pass
        progress_bar.progress((i + 1) / limit, text=progress_text)
        
    progress_bar.empty()
    
    df = pd.DataFrame(stories_data)
    if not df.empty:
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        df['sentiment_score'] = df['title'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        
        def categorize(score):
            if score > 0.05: return "Positive"
            elif score < -0.05: return "Negative"
            else: return "Neutral"
            
        df['sentiment_category'] = df['sentiment_score'].apply(categorize)
    return df

# Word list generator for trending keywords
def get_top_keywords(titles, top_n=15):
    # Standard English stop words + common Tech/HN specific filler words
    stop_words = set([
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
        'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
        'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres',
        'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is',
        'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of',
        'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
        'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats',
        'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll',
        'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt',
        'we', 'wed', 'well', 'were', 'weve', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which',
        'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll',
        'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves', 'show', 'hn', 'ask', 'tell', 'yc', 'new', 'post',
        'use', 'using', 'get', 'make', 'made', 'like', 'one', 'two', 'first', 'day', 'year', 'years'
    ])
    words = []
    for title in titles:
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', str(title)).lower()
        for word in cleaned.split():
            if word and word not in stop_words and len(word) > 2:
                words.append(word)
    return Counter(words).most_common(top_n)


# Sidebar controls
st.sidebar.markdown("### 🛠️ Configuration")
limit = st.sidebar.slider("Stories to Fetch", min_value=10, max_value=100, value=50, step=10)

# Fetch Data
df_raw = fetch_and_analyze(limit)

if df_raw.empty:
    st.warning("⚠️ No data fetched. Please check your internet connection or HN API status.")
else:
    # Sidebar Filters (applied to fetched data to keep interactions fast)
    max_upvotes_possible = int(df_raw['score'].max()) if len(df_raw) > 0 else 500
    st.sidebar.markdown("### 🔍 Filters")
    min_score = st.sidebar.slider("Minimum Upvote Score", min_value=0, max_value=max_upvotes_possible, value=0, step=5)
    selected_sentiment = st.sidebar.multiselect(
        "Sentiment Category", 
        options=["Positive", "Neutral", "Negative"], 
        default=["Positive", "Neutral", "Negative"]
    )
    
    if st.sidebar.button("🔄 Refresh Cache Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Filter Dataframe
    df = df_raw[
        (df_raw['score'] >= min_score) & 
        (df_raw['sentiment_category'].isin(selected_sentiment))
    ]

    # Metrics Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Stories Filtered</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_upvotes = round(df['score'].mean()) if len(df) > 0 else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔥</div>
                <div class="metric-value">{avg_upvotes}</div>
                <div class="metric-label">Average Upvotes</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        mean_sentiment = df['sentiment_score'].mean() if len(df) > 0 else 0
        if mean_sentiment > 0.05:
            vibe, vibe_icon = "Positive", "🟢"
        elif mean_sentiment < -0.05:
            vibe, vibe_icon = "Negative", "🔴"
        else:
            vibe, vibe_icon = "Neutral", "⚪"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{vibe_icon}</div>
                <div class="metric-value">{vibe}</div>
                <div class="metric-label">Overall Sentiment</div>
            </div>
        """, unsafe_allow_html=True)

    # Main Tabs Layout
    tab_analytics, tab_explorer, tab_trends = st.tabs([
        "📈 Analytics Dashboard", 
        "📰 Headline Explorer", 
        "🏷️ Keyword Trends & Raw Data"
    ])

    # TAB 1: ANALYTICS DASHBOARD
    with tab_analytics:
        if df.empty:
            st.info("No data matches current filters. Adjust your filters in the sidebar.")
        else:
            chart_col1, chart_col2 = st.columns([2, 1])
            
            with chart_col1:
                # Scatter plot: Upvotes vs. Sentiment
                fig_scatter = px.scatter(
                    df, 
                    x='sentiment_score', 
                    y='score', 
                    hover_name='title',
                    color='sentiment_category',
                    color_discrete_map={"Positive": "#10B981", "Neutral": "#9CA3AF", "Negative": "#EF4444"},
                    labels={'sentiment_score': 'Headline Sentiment Score', 'score': 'Upvotes'},
                    title='Headline Sentiment vs. Upvote Score',
                    template='plotly_dark'
                )
                fig_scatter.update_traces(
                    marker=dict(size=12, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)'))
                )
                fig_scatter.update_layout(
                    height=450,
                    margin=dict(t=50, b=30, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with chart_col2:
                # Donut Chart: Sentiment distribution
                sentiment_counts = df['sentiment_category'].value_counts().reset_index()
                sentiment_counts.columns = ['sentiment_category', 'count']
                
                fig_donut = px.pie(
                    sentiment_counts, 
                    values='count', 
                    names='sentiment_category', 
                    hole=0.45,
                    color='sentiment_category',
                    color_discrete_map={"Positive": "#10B981", "Neutral": "#9CA3AF", "Negative": "#EF4444"},
                    title="Sentiment Breakdown",
                    template="plotly_dark"
                )
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                fig_donut.update_layout(
                    height=450,
                    showlegend=False, 
                    margin=dict(t=50, b=20, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_donut, use_container_width=True)

    # TAB 2: HEADLINE EXPLORER
    with tab_explorer:
        search_query = st.text_input("🔍 Search headlines...", "", placeholder="Type keywords to filter stories...")
        
        df_expl = df[df['title'].str.contains(search_query, case=False)] if search_query else df
        
        if df_expl.empty:
            st.info("No headlines match the search term.")
        else:
            st.write(f"Showing **{len(df_expl)}** matching headlines:")
            for idx, row in df_expl.iterrows():
                sentiment = row['sentiment_category']
                score = row['score']
                title = row['title']
                url = row['url'] if row['url'] else f"https://news.ycombinator.com/item?id={row.get('timestamp')}"
                
                badge_class = "badge-positive" if sentiment == "Positive" else "badge-negative" if sentiment == "Negative" else "badge-neutral"
                
                st.markdown(f"""
                    <div class="story-card">
                        <div class="story-header">
                            <a class="story-title" href="{url}" target="_blank">{title}</a>
                        </div>
                        <div class="story-meta">
                            <span class="badge {badge_class}">{sentiment}</span>
                            <span class="badge badge-score">🔥 {score} upvotes</span>
                            <span style="font-size: 0.8rem;">⏰ {row['time'].strftime('%Y-%m-%d %H:%M')}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # TAB 3: KEYWORD TRENDS & RAW DATA
    with tab_trends:
        if df.empty:
            st.info("No data available to analyze keywords.")
        else:
            keyword_col, table_col = st.columns([1, 1])
            
            with keyword_col:
                top_words = get_top_keywords(df['title'])
                if top_words:
                    df_words = pd.DataFrame(top_words, columns=['Word', 'Count'])
                    fig_words = px.bar(
                        df_words,
                        x='Count',
                        y='Word',
                        orientation='h',
                        title='Trending Headline Keywords (Stop-words removed)',
                        template='plotly_dark',
                        color='Count',
                        color_continuous_scale='Bluered'
                    )
                    fig_words.update_layout(
                        height=450,
                        showlegend=False,
                        coloraxis_showscale=False,
                        yaxis={'categoryorder': 'total ascending'},
                        margin=dict(t=50, b=30, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                else:
                    st.info("Not enough keywords found.")
            
            with table_col:
                st.write("### 🗃️ Complete Dataset")
                st.dataframe(
                    df[['title', 'score', 'sentiment_category', 'sentiment_score', 'time']],
                    use_container_width=True,
                    height=450
                )