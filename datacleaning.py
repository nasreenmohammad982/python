import pandas as pd
import nltk
from nltk.corpus import stopwords

print("Starting data cleaning...")

# Load Dataset
df = pd.read_csv('c:\Users\HP\Downloads\bbc_news.csv')   # Use local file path

# Basic Info
print(df.info())
print(df.isnull().sum())

# Remove missing & duplicate values
df = df.dropna()
df = df.drop_duplicates()

# Convert to lowercase
df["text"] = df["text"].str.lower()

# Remove non-alphabet characters
df["text"] = df["text"].str.replace('[^a-z ]', ' ', regex=True)

# Remove stopwords
nltk.download("stopwords")
stop = set(stopwords.words("english"))
df["text"] = df["text"].apply(lambda x: " ".join([w for w in x.split() if w not in stop]))

# Save cleaned dataset
df.to_csv('c:\Users\HP\Downloads\bbc_news.csv', index=False)

print("Cleaning complete!")
