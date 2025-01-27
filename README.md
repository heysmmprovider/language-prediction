# Language Detection with XLM-RoBERTa

This is a Flask-based API for detecting languages in text using the pre-trained `papluca/xlm-roberta-base-language-detection` model. It leverages **PyTorch** and **Hugging Face Transformers** for efficient and accurate language classification.

<img src="https://cdn.mypanel.link/h1sa68/zwquifzpzxn5h5tc.png" alt="HeySmmReseller Logo" style="width: 100%;">

## Features

- **Multi-Language Support**: Detects 20 languages including English, Arabic, Chinese, Spanish, and more.
- **Top Predictions**: Provides the top 3 language predictions with confidence scores and entropy.
- **Efficient Deployment**: Supports GPU acceleration with CUDA if available for faster inference.
- **Configurable and Ready to Use**: Available as a Docker image for easy deployment.

## Supported Languages

The following languages are supported:

| Language | Code | Language | Code |
|----------|------|----------|------|
| Arabic   | `ar` | Italian  | `it` |
| Bulgarian| `bg` | Japanese | `ja` |
| German   | `de` | Dutch    | `nl` |
| Greek    | `el` | Polish   | `pl` |
| English  | `en` | Portuguese| `pt` |
| Spanish  | `es` | Russian  | `ru` |
| French   | `fr` | Swahili  | `sw` |
| Hindi    | `hi` | Thai     | `th` |
| Turkish  | `tr` | Urdu     | `ur` |
| Vietnamese| `vi`| Chinese  | `zh` |

## Quick Start

### Use the Docker Image

Pull and run the Docker container:

```bash
docker pull heysmmprovider/language-prediction
docker run --gpus all -p 10005:10005 heysmmprovider/language-prediction
```

### API Usage

1. **Endpoint**: The API listens on `http://<host>:10005/`.
2. **Request**: Send a POST request with JSON data containing the text to analyze:
   ```json
   {
     "text": "Bonjour! Comment ça va?"
   }
   ```
3. **Response**: The API will return the top 3 predicted languages with confidence scores and entropy:
   ```json
   {
     "predictions": [
       {
         "language": "fr",
         "confidence": 0.98,
         "entropy": 0.025
       },
       {
         "language": "en",
         "confidence": 0.01,
         "entropy": 0.025
       },
       {
         "language": "es",
         "confidence": 0.01,
         "entropy": 0.025
       }
     ],
     "input_length": 20
   }
   ```

### Environment Variable for Port Configuration

You can configure the API to listen on a specific port using the `LISTEN_PORT` environment variable. By default, it uses port `10005`.

## About Us

**heysmmprovider** is a trusted brand under **heysmmreseller**, delivering scalable and high-performance solutions in machine learning and social media management. 

Explore more at **[heysmmprovider.com](#)** and transform your language processing tasks into seamless operations.

---

For inquiries or issues, feel free to contact our support team.

**[heysmmreseller.com](#)** 
