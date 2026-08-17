import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")


def predict(transaction_id):
    try:
        response = requests.post(
            API_URL,
            json={"transaction_id": int(transaction_id)},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result["prediction"], result["fraud_probability"], result["explanation"]
    except Exception as exc:
        return "error", 0.0, {"message": str(exc)}


with gr.Blocks(title="Graph Fraud AI") as demo:
    gr.Markdown("# Graph Fraud AI\nTransaction-level fraud investigation demo.")
    transaction_id = gr.Number(value=0, precision=0, label="Transaction ID")
    button = gr.Button("Analyze", variant="primary")
    prediction = gr.Textbox(label="Decision")
    probability = gr.Number(label="Fraud probability")
    explanation = gr.JSON(label="Explanation")
    button.click(predict, inputs=transaction_id, outputs=[prediction, probability, explanation])


if __name__ == "__main__":
    demo.launch()
