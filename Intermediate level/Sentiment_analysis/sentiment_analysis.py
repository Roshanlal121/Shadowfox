import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import os

# =========================
# LOAD DATASET
# =========================

FILE_NAME = "X data.csv"

try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print("❌ ERROR: CSV file not found!")
    print("👉 Make sure 'X data.csv' is in the SAME folder as this script")
    exit()

print("\n✅ Dataset Loaded Successfully\n")
print(df.head())

# =========================
# CHECK COLUMN
# =========================

if "clean_text" not in df.columns:
    print("❌ ERROR: 'clean_text' column missing in dataset")
    exit()

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
# APPLY SENTIMENT
# =========================

df["Sentiment"] = df["clean_text"].apply(get_sentiment)

# =========================
# CREATE FOLDERS
# =========================

os.makedirs("graphs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# =========================
# SENTIMENT COUNT
# =========================

print("\n📊 Sentiment Counts:\n")
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

counts = df["Sentiment"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
plt.title("Sentiment Percentage")
plt.savefig("graphs/pie_chart.png")
plt.show()

# =========================
# WORDCLOUD - POSITIVE
# =========================

positive_text = " ".join(df[df["Sentiment"] == "Positive"]["clean_text"].astype(str))

if positive_text.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(positive_text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Positive WordCloud")
    plt.savefig("graphs/positive_wordcloud.png")
    plt.show()

# =========================
# WORDCLOUD - NEGATIVE
# =========================

negative_text = " ".join(df[df["Sentiment"] == "Negative"]["clean_text"].astype(str))

if negative_text.strip():
    wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Negative WordCloud")
    plt.savefig("graphs/negative_wordcloud.png")
    plt.show()

# =========================
# SAVE OUTPUT
# =========================

df.to_csv("output/sentiment_output.csv", index=False)

print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("📁 Check 'graphs/' folder")
print("📁 Check 'output/' folder")