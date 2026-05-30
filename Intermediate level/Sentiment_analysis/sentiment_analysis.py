import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# =========================
# CREATE REQUIRED FOLDERS
# =========================

os.makedirs("graphs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# =========================
# LOAD DATASET (ZIP FILE)
# =========================

df = pd.read_csv("data/X data.zip", compression='zip')

print("\nDataset Preview:\n")
print(df.head())

# =========================
# SENTIMENT FUNCTION
# =========================

def get_sentiment(text):
    analysis = TextBlob(str(text))

    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# =========================
# APPLY SENTIMENT ANALYSIS
# =========================

df["Sentiment"] = df["clean_text"].apply(get_sentiment)

# =========================
# SENTIMENT COUNTS
# =========================

print("\nSentiment Counts:\n")
print(df["Sentiment"].value_counts())

# =========================
# BAR GRAPH
# =========================

plt.figure(figsize=(6,5))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")

plt.savefig("graphs/bar_graph.png")
plt.show()

# =========================
# PIE CHART
# =========================

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(7,7))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
plt.title("Sentiment Percentage")

plt.savefig("graphs/pie_chart.png")
plt.show()

# =========================
# POSITIVE WORD CLOUD
# =========================

positive_text = " ".join(df[df["Sentiment"] == "Positive"]["clean_text"])

if positive_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Positive Tweets WordCloud")

    plt.savefig("graphs/positive_wordcloud.png")
    plt.show()

# =========================
# NEGATIVE WORD CLOUD
# =========================

negative_text = " ".join(df[df["Sentiment"] == "Negative"]["clean_text"])

if negative_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Negative Tweets WordCloud")

    plt.savefig("graphs/negative_wordcloud.png")
    plt.show()

# =========================
# SAVE OUTPUT CSV
# =========================

df.to_csv("output/sentiment_output.csv", index=False)

print("\nProject Completed Successfully ")
print("Graphs saved inside 'graphs' folder")
print("CSV saved inside 'output' folder")
