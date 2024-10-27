# Required Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Load the WhatsApp data (Make sure to replace this with your actual data)
file_path = "WhatsApp Chat with MTech AIML IITJ (24-26).txt"  # Replace with the path to your file

# Step 1: Load and Parse WhatsApp Data
import re
from datetime import datetime

with open(file_path, "r", encoding="utf-8") as file:
    chat_data = file.readlines()

# Regular expression to match the timestamp and message format
pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s(?:AM|PM|am|pm))\s-\s(.?):\s(.)"
data = []

# Parsing the data into structured format
for line in chat_data:
    match = re.match(pattern, line.strip())
    if match:
        date_str, time_str, sender, message = match.groups()
        timestamp_str = f"{date_str}, {time_str}"
        timestamp = datetime.strptime(timestamp_str, "%d/%m/%y, %I:%M %p")
        data.append([timestamp, sender, message])

# Create DataFrame from parsed data
df = pd.DataFrame(data, columns=["timestamp", "sender", "message"])

# Step 2: Use PandasAI with SmartDataframe
# Set API key for OpenAI or other LLM in the environment variable
os.environ["PANDASAI_API_KEY"] = ""  # Replace with your actual OpenAI API key

# Use SmartDataframe from PandasAI
df_smart = SmartDataframe(df)

# Example Prompt: Get insights on the most active participants and their sentiment
response = df_smart.chat(
    "Who are the most active participants and what is the overall sentiment they express?"
)
print(response)

# Step 3: Sentiment Analysis (with TextBlob, for example)
from textblob import TextBlob


def get_sentiment(message):
    analysis = TextBlob(message)
    return analysis.sentiment.polarity


df["sentiment"] = df["message"].apply(get_sentiment)
df["sentiment_label"] = df["sentiment"].apply(
    lambda x: "positive" if x > 0 else ("negative" if x < 0 else "neutral")
)

# Step 4: Advanced Visualizations

# 4.1 Heatmap of message activity by hour and day
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()

# Creating a pivot table to show message count by hour and day
activity_matrix = df.pivot_table(
    index="weekday", columns="hour", values="message", aggfunc="count"
).fillna(0)

plt.figure(figsize=(12, 8))
sns.heatmap(activity_matrix, cmap="YlGnBu", annot=True)
plt.title("Heatmap of Message Activity by Hour and Day of Week")
plt.show()

# 4.2 Word Cloud for frequent words
text = " ".join(df["message"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Frequent Words in Group Chat")
plt.show()

# 4.3 Sentiment Distribution Bar Plot
sentiment_count = df["sentiment_label"].value_counts()

plt.figure(figsize=(7, 5))
sentiment_count.plot(kind="bar", color=["green", "red", "gray"])
plt.title("Sentiment Distribution in Group Chat")
plt.xlabel("Sentiment")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.show()

# 4.4 Daily Message Activity Line Plot
df["date"] = df["timestamp"].dt.date
daily_activity = df.groupby("date").size()

plt.figure(figsize=(10, 6))
daily_activity.plot(kind="line", color="purple")
plt.title("Daily Message Activity")
plt.xlabel("Date")
plt.ylabel("Number of Messages")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.5 Weekly Message Activity Bar Plot
weekly_activity = (
    df.groupby("weekday")
    .size()
    .reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
)

plt.figure(figsize=(10, 6))
weekly_activity.plot(kind="bar", color="teal")
plt.title("Weekly Message Activity")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Messages")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
