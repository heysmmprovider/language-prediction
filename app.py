from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = Flask(__name__)

# Initialize model and tokenizer
MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

# Get language mapping directly from model config
id2lang = model.config.id2label

def predict_language(text):
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get top 3 predictions
        values, predictions = torch.topk(probabilities, k=3)
        
        results = []
        entropy = float(-(probabilities * torch.log(probabilities + 1e-9)).sum())
        
        for prob, pred in zip(values[0], predictions[0]):
            lang = id2lang[pred.item()]
            results.append({
                "language": lang,
                "confidence": float(prob),
                "entropy": entropy
            })
            
        return results

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return None

@app.route("/", methods=["POST"])
def detect_language():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid request format"}), 400

        text = data['text']
        if not isinstance(text, str):
            return jsonify({"error": "Text must be a string"}), 400

        predictions = predict_language(text)
        if predictions is None:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({
            "predictions": predictions,
            "input_length": len(text)
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info(f"Using device: {device}")
    logging.info(f"Available languages: {list(id2lang.values())}")
    port = int(os.getenv("LISTEN_PORT", 10005))
    app.run(host='0.0.0.0', port)
