# Deployment Guide

Guide to deploying and using your fine-tuned model in production.

## Understanding Deployment

**Key Concept:** After training, your model is **self-contained** and doesn't need access to your codebase.

See [Training vs Deployment](../training-vs-deployment.md) for details.

## Model Files

After training, you'll have:

```
./models/fine-tuned/
├── adapter_config.json    # LoRA configuration
├── adapter_model.bin      # Trained weights (50-200MB)
├── tokenizer.json         # Tokenizer
├── tokenizer_config.json
└── special_tokens_map.json
```

**This is everything you need for deployment!**

## Deployment Options

### Option 1: Python API

Use the model directly in Python applications.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

# Ask a question
prompt = """### Instruction:
Which module handles user authentication?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Option 2: REST API Server

Create a Flask/FastAPI server:

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load model at startup
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    
    prompt = f"""### Instruction:
{question}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does login work?"}'
```

### Option 3: Ollama Integration

Deploy with Ollama for local inference:

```bash
# 1. Create Modelfile
cat > Modelfile << EOF
FROM deepseek-coder:6.7b
ADAPTER ./models/fine-tuned/adapter_model.bin
EOF

# 2. Create custom model
ollama create my-code-expert -f Modelfile

# 3. Use it
ollama run my-code-expert
```

### Option 4: Chatbot Integration

#### With LangChain

```python
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=500
)

# Create LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create conversation chain
conversation = ConversationChain(llm=llm)

# Use it
response = conversation.run("Which module handles authentication?")
print(response)
```

## Production Considerations

### 1. Model Loading Optimization

**Cache the model:**
```python
import functools

@functools.lru_cache(maxsize=1)
def load_model():
    model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
    tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")
    return model, tokenizer

# Use cached version
model, tokenizer = load_model()
```

### 2. GPU Configuration

**Specify GPU:**
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# For multi-GPU
model = torch.nn.DataParallel(model)
```

### 3. Batching Requests

```python
def generate_batch(questions, model, tokenizer):
    prompts = [f"### Instruction:\n{q}\n\n### Response:\n" for q in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=500)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses
```

### 4. Response Streaming

```python
from transformers import TextIteratorStreamer
from threading import Thread

def stream_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer)
    
    # Generate in background thread
    generation_kwargs = dict(inputs, streamer=streamer, max_length=500)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream output
    for text in streamer:
        yield text
```

## Optimizations

### Convert to GGUF (llama.cpp)

For CPU inference:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert model
python convert.py ./models/fine-tuned --outfile model.gguf

# Quantize (optional, for smaller size)
./quantize model.gguf model-q4_0.gguf q4_0

# Run
./main -m model-q4_0.gguf -p "Which module handles auth?"
```

### ONNX Export

For optimized inference:

```python
from optimum.onnxruntime import ORTModelForCausalLM

# Export to ONNX
model = ORTModelForCausalLM.from_pretrained(
    "./models/fine-tuned",
    export=True
)
model.save_pretrained("./models/onnx")

# Use ONNX model
model = ORTModelForCausalLM.from_pretrained("./models/onnx")
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model
COPY ./models/fine-tuned ./models/fine-tuned

# Copy server code
COPY server.py .

# Expose port
EXPOSE 5000

# Run server
CMD ["python", "server.py"]
```

### Build and Run

```bash
# Build image
docker build -t my-code-expert .

# Run container
docker run -p 5000:5000 my-code-expert

# With GPU support
docker run --gpus all -p 5000:5000 my-code-expert
```

## Monitoring

### Log Requests

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    logger.info(f"Question received: {question}")
    
    response = generate_response(question)
    logger.info(f"Response generated: {len(response)} chars")
    
    return jsonify({'answer': response})
```

### Track Performance

```python
import time

@app.route('/ask', methods=['POST'])
def ask_question():
    start_time = time.time()
    
    question = request.json['question']
    response = generate_response(question)
    
    duration = time.time() - start_time
    logger.info(f"Request took {duration:.2f}s")
    
    return jsonify({
        'answer': response,
        'duration_ms': duration * 1000
    })
```

## Security

### API Authentication

```python
from functools import wraps
from flask import request, abort

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            abort(401)
        return f(*args, **kwargs)
    return decorated_function

@app.route('/ask', methods=['POST'])
@require_api_key
def ask_question():
    # ... your code
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/ask', methods=['POST'])
@limiter.limit("10 per minute")
def ask_question():
    # ... your code
```

## Scaling

### Horizontal Scaling

Use load balancer with multiple model instances:

```
          Load Balancer
         /      |      \
    Server1  Server2  Server3
      |        |        |
    Model    Model    Model
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_response(question):
    return generate_response(question)
```

## Updating the Model

When your codebase changes:

```bash
# 1. Re-analyze and train
python main.py

# 2. Replace model in production
cp -r ./models/fine-tuned /production/models/fine-tuned

# 3. Restart service
systemctl restart code-expert-api
```

## Next Steps

- **Error Debugging**: [Debugging Guide](debugging.md)
- **Troubleshooting**: [Troubleshooting Guide](../reference/troubleshooting.md)
- **Examples**: Check `examples/` directory
