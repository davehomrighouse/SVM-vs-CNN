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

    return label, confidence


def run_comparison(text):
    text = text.strip()
    if not text:
        return "", "", "", ""

    results = compare_models(text)

    return (
        results["SVM Prediction"],
        results["SVM Confidence"],
        results["CNN Prediction"],
        results["CNN Confidence"]
    )


with gr.Blocks() as demo:
    gr.Markdown("# Reuters Text Classification Demo")
    gr.Markdown(
        "Compare predictions from a TF-IDF + SVM model and a CNN model trained on the Reuters dataset."
    )

    shared_input = gr.Textbox(
        label="Enter text",
        lines=6,
        placeholder="Type or paste text here..."
    )

    with gr.Tab("Single Model"):
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
            inputs=[shared_input, model_choice],
            outputs=[pred_label, pred_conf]
        )

    with gr.Tab("Compare Models"):
        compare_btn = gr.Button("Compare")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## SVM")
                svm_pred = gr.Textbox(label="Predicted Category")
                svm_conf = gr.Textbox(label="Probability/Tier", lines=4)

            with gr.Column():
                gr.Markdown("## CNN")
                cnn_pred = gr.Textbox(label="Predicted Category")
                cnn_conf = gr.Textbox(label="Probability/Tier", lines=5)

        compare_btn.click(
            fn=run_comparison,
            inputs=shared_input,
            outputs=[svm_pred, svm_conf, cnn_pred, cnn_conf]
        )

demo.launch()
