import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load artifacts once at startup
class_labels = joblib.load("reuters_class_labels.pkl")

svm_model = joblib.load("models/svm_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

cnn_model = load_model("models/cnn_model.keras")
tokenizer = joblib.load("models/tokenizer.pkl")
maxlen = joblib.load("models/maxlen.pkl")


def confidence_band(conf_percent):
    if 0 <= conf_percent < 40:
        return "Weak Signal"
    elif 40 <= conf_percent < 80:
        return "Moderate Signal"
    elif 80 <= conf_percent <= 100:
        return "Strong Signal"
    return "Unknown"
    

def predict_svm(text: str):
    text_vec = tfidf_vectorizer.transform([text])
    pred_idx = svm_model.predict(text_vec)[0]
    pred_label = class_labels[pred_idx]

    if hasattr(svm_model, "predict_proba"):
        probs = svm_model.predict_proba(text_vec)[0]
        confidence = float(probs[pred_idx])
        conf_percent = confidence * 100
        band = confidence_band(conf_percent)
        confidence_text = f"{conf_percent:.2f}% ({band})"
    else:
        confidence_text = "N/A"

    return pred_label, confidence_text
    

def predict_cnn(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)

    probs = cnn_model.predict(padded, verbose=0)[0]
    top3_index = probs.argsort()[-3:][::-1]

    best_index = top3_index[0]
    best_label = class_labels[best_index]

    lines = []
    for idx in top3_index:
        label = class_labels[idx]
        conf_percent = float(probs[idx] * 100)
        band = confidence_band(conf_percent)

        lines.append(
            f"{label}: {conf_percent:.2f}% ({band})"
        )

    return best_label, "\n".join(lines)


def compare_models(text: str):
    svm_label, svm_conf = predict_svm(text)
    cnn_label, cnn_conf = predict_cnn(text)

    return {
        "SVM Prediction": svm_label,
        "SVM Confidence": svm_conf,
        "CNN Prediction": cnn_label,
        "CNN Confidence": cnn_conf
    }
