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

# Language codes mapping
LANGUAGE_CODES = {
    'LABEL_0': 'ar', 'LABEL_1': 'bg', 'LABEL_2': 'de', 'LABEL_3': 'el', 
    'LABEL_4': 'en', 'LABEL_5': 'es', 'LABEL_6': 'fr', 'LABEL_7': 'hi', 
    'LABEL_8': 'it', 'LABEL_9': 'ja', 'LABEL_10': 'nl', 'LABEL_11': 'pl', 
    'LABEL_12': 'pt', 'LABEL_13': 'ru', 'LABEL_14': 'sw', 'LABEL_15': 'th', 
    'LABEL_16': 'tr', 'LABEL_17': 'ur', 'LABEL_18': 'vi', 'LABEL_19': 'zh'
}

def predict_language(text):
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Get predictions and confidences
        values, predictions = torch.topk(probabilities, k=3)
        
        results = []
        entropy = float(-(probabilities * torch.log(probabilities + 1e-9)).sum())
        
        for prob, pred in zip(values[0], predictions[0]):
            lang_code = LANGUAGE_CODES[f'LABEL_{pred.item()}']
            results.append({
                "language": lang_code,
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
    app.run(host='0.0.0.0', port=10005)