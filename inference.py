import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load artifacts once at startup
svm_model = joblib.load("models/svm_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

cnn_model = load_model("models/cnn_model.keras")
tokenizer = joblib.load("models/tokenizer.pkl")
maxlen = joblib.load("models/maxlen.pkl")


def predict_svm(text: str):
    text_vec = tfidf_vectorizer.transform([text])
    pred_idx = svm_model.predict(text_vec)[0]

    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(text_vec)[0]
        confidence = float(np.max(probs))
    else:
        confidence = None

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_label, confidence


def predict_cnn(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

    probs = cnn_model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_label, confidence


def compare_models(text: str):
    svm_label, svm_conf = predict_svm(text)
    cnn_label, cnn_conf = predict_cnn(text)

    result = {
        "SVM Prediction": svm_label,
        "SVM Confidence": round(svm_conf, 4) if svm_conf is not None else "N/A",
        "CNN Prediction": cnn_label,
        "CNN Confidence": round(cnn_conf, 4)
    }
    return result
