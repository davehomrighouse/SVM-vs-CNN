import gradio as gr
from inference import predict_svm, predict_cnn, compare_models

def run_single_model(text, model_choice):
    text = text.strip()
    if not text:
        return "Please enter some text.", ""

    if model_choice == "SVM":
        label, confidence = predict_svm(text)
    else:
        label, confidence = predict_cnn(text)

    confidence_text = f"{confidence:.4f}" if isinstance(confidence, float) else "N/A"
    return label, confidence_text


def run_comparison(text):
    text = text.strip()
    if not text:
        return {
            "SVM Prediction": "",
            "SVM Confidence": "",
            "CNN Prediction": "",
            "CNN Confidence": ""
        }

    return compare_models(text)


with gr.Blocks() as demo:
    gr.Markdown("# Reuters Text Classification Demo")
    gr.Markdown(
        "Compare predictions from a TF-IDF + SVM model and a CNN model trained on the Reuters dataset."
    )

    with gr.Tab("Single Model"):
        text_input = gr.Textbox(
            label="Enter news text",
            lines=6,
            placeholder="Type or paste a short news article snippet here..."
        )
        model_choice = gr.Radio(
            ["SVM", "CNN"],
            value="SVM",
            label="Choose model"
        )
        predict_btn = gr.Button("Predict")
        pred_label = gr.Textbox(label="Predicted Category")
        pred_conf = gr.Textbox(label="Confidence")

        predict_btn.click(
            fn=run_single_model,
            inputs=[text_input, model_choice],
            outputs=[pred_label, pred_conf]
        )

    with gr.Tab("Compare Models"):
        compare_input = gr.Textbox(
            label="Enter news text",
            lines=6,
            placeholder="Paste the same input here to compare both models..."
        )
        compare_btn = gr.Button("Compare")
        compare_output = gr.JSON(label="Model Comparison")

        compare_btn.click(
            fn=run_comparison,
            inputs=compare_input,
            outputs=compare_output
        )

demo.launch()
