import pandas as pd
import re
import string

def load_hindi_stopwords(filepath='/content/drive/MyDrive/stopwords-hi.txt'):
    with open(filepath, encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

hindi_stop_words = load_hindi_stopwords()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets
    # Remove punctuation: combine Python's punctuation with Hindi danda "।" and double danda "॥"
    punctuation = string.punctuation + "।" + "॥"
    text = re.sub(f"[{re.escape(punctuation)}]", '', text)
    # Remove stopwords
    words = [word for word in text.split() if word not in hindi_stop_words]
    return " ".join(words)

def preprocess_data(input_csv='/content/drive/MyDrive/songshappy_sad.csv', output_csv='/content/drive/MyDrive/songs_happysadcleaned.csv'):
    df = pd.read_csv(input_csv)
    df.dropna(subset=['Lyrics', 'Genre'], inplace=True)
    df['cleaned_lyrics'] = df['Lyrics'].apply(clean_text)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == '__main__':
    preprocess_data()
