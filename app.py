from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Set the maximum sequence length (common for both models)
MAX_SEQUENCE_LENGTH = 427

def load_model_artifacts(model_path, tokenizer_path, label_map_path):
    """Load model, tokenizer, and label map; return model and reverse label map."""
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
    reverse_label_map = {v: k for k, v in label_map.items()}
    return model, tokenizer, reverse_label_map

# Load Hindi model artifacts
hindi_model, hindi_tokenizer, hindi_reverse_label_map = load_model_artifacts(
    'lstm_genre_model_hindi.h5',
    'tokenizer_hindi.pkl',
    'label_map_hindi.pkl'
)

# Load English model artifacts
english_model, english_tokenizer, english_reverse_label_map = load_model_artifacts(
    'lstm_genre_modelbest.h5',
    'tokenizerbest.pkl',
    'label_mapbest.pkl'
)

def predict_genre(lyrics, model, tokenizer, reverse_label_map):
    """Predict the single most likely genre (without probability)."""
    lyrics = lyrics.lower()
    sequence = tokenizer.texts_to_sequences([lyrics])
    padded_seq = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded_seq)[0]
    top_index = np.argmax(pred)
    genre = reverse_label_map.get(top_index, "Unknown")
    return genre

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        language = request.form.get('language')
        lyrics = request.form.get('lyrics', '').strip()
        if not lyrics:
            result = "Please enter some lyrics."
        else:
            if language == 'english':
                result = predict_genre(lyrics, english_model, english_tokenizer, english_reverse_label_map)
            elif language == 'hindi':
                result = predict_genre(lyrics, hindi_model, hindi_tokenizer, hindi_reverse_label_map)
            else:
                result = "Invalid language selection."
        # If AJAX, return JSON response
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'result': result})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
