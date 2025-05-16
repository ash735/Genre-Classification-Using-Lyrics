import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuation
    words = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    return " ".join(words)

def preprocess_data(input_csv='/content/drive/MyDrive/data_genre.csv', output_csv='/content/drive/MyDrive/lyrics_cleaneddd.csv'):
    # Load the dataset (expects columns 'Genre' and 'Lyrics')
    df = pd.read_csv(input_csv)
    # Drop rows missing either Lyrics or Genre
    df.dropna(subset=['Lyrics', 'Genre'], inplace=True)
    # Clean the Lyrics column
    df['cleaned_lyrics'] = df['Lyrics'].apply(clean_text)
    # Save the cleaned data
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == '__main__':
    preprocess_data()
