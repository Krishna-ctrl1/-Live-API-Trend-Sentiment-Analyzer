import requests
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

print("Fetching live data from Hacker News...")

top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
top_story_ids = requests.get(top_stories_url).json()


stories_data = []
for story_id in top_story_ids[:50]:
    story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
    story = requests.get(story_url).json()
    
    # Safety check: some API items might be empty or deleted
    if story and 'title' in story:
        stories_data.append({
            'title': story.get('title'),
            'score': story.get('score', 0),
            'url': story.get('url', ''),
            'timestamp': story.get('time')
        })

df = pd.DataFrame(stories_data)

df['time'] = pd.to_datetime(df['timestamp'], unit='s')


df['sentiment_score'] = df['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

def categorize_sentiment(score):
    if score > 0.05: return "Positive"
    elif score < -0.05: return "Negative"
    else: return "Neutral"

df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)

print("\nData Processing Complete. Generating Dashboard...")

fig = px.scatter(
    df,
    x='sentiment_score',
    y='score',
    hover_name='title',
    color='sentiment_category',
    color_discrete_map={"Positive": "#00CC96", "Neutral": "#888888", "Negative": "#EF553B"},
    title='Live Hacker News Trends: Upvotes vs. Headline Sentiment',
    labels={'sentiment_score': 'Sentiment (-1.0 to 1.0)', 'score': 'Upvote Score'},
    template='plotly_dark'
)

fig.show()