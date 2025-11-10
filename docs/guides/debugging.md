# Error Debugging Guide

Use your fine-tuned model to debug errors and identify responsible modules.

## Key Concept

**After training, the model can identify error sources WITHOUT accessing your codebase.**

The model learned your code structure during training and can answer from memory.

See [Training vs Deployment](../training-vs-deployment.md) for how this works.

## Use Cases

### 1. Identify Error Source

**Question:**
```
I'm seeing: "AttributeError: 'NoneType' object has no attribute 'process_data'"
Which module is responsible?
```

**Model Response:**
```
Check the data processing modules:
- data_processor.py - Main data processing logic
- utils/data_handler.py - Helper functions

The error suggests an object with 'process_data' method is None.
Likely causes:
1. Data processor wasn't initialized
2. Function returned None instead of processor object
```

### 2. Traceback Analysis

**Question:**
```
Traceback:
  File "src/api/endpoints.py", line 45, in create_user
    user = UserManager.create(data)
  File "src/auth/manager.py", line 78, in create
    return self.repository.save(user)
AttributeError: 'NoneType' object has no attribute 'save'

Which module is the issue?
```

**Model Response:**
```
The error is in src/auth/manager.py - self.repository is None.

Check:
1. auth/manager.py - UserManager.__init__() method
2. api/endpoints.py - How UserManager is instantiated
3. Ensure repository dependency is properly injected
```

### 3. Error Pattern Recognition

**Question:**
```
I keep getting "Connection timeout" errors.
Which modules handle database connections?
```

**Model Response:**
```
Database connection modules:
1. db/connection.py - Connection pool and configuration
2. db/session.py - Session management
3. config/database.py - Database settings

For timeouts:
- Check connection pool size in db/connection.py
- Look for unclosed sessions in db/session.py
- Review timeout settings
```

## Python Integration

### Basic Error Debugging

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

def debug_error(error_message):
    prompt = f"""### Instruction:
I'm seeing this error: {error_message}
Which module would be responsible for this?

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Use it
error = "AttributeError: 'NoneType' object has no attribute 'process'"
analysis = debug_error(error)
print(analysis)
```

### Advanced Analysis

```python
def analyze_traceback(traceback_text):
    prompt = f"""### Instruction:
Analyze this traceback and identify the root cause:

{traceback_text}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=700, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# Example
traceback = """
Traceback (most recent call last):
  File "api/routes.py", line 123, in process_order
    result = payment_service.charge(amount)
  File "services/payment.py", line 45, in charge
    return self.gateway.process(data)
TypeError: 'NoneType' object is not callable
"""

analysis = analyze_traceback(traceback)
print(analysis)
```

## Building a Debugging Assistant

### Flask Server

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load model at startup
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

@app.route('/debug', methods=['POST'])
def debug_error():
    error = request.json['error']
    
    prompt = f"""### Instruction:
I'm seeing this error: {error}
Which module would be responsible?

### Response:
"""
    
    response = generate_response(prompt)
    return jsonify({'analysis': response})

@app.route('/find-module', methods=['POST'])
def find_module():
    functionality = request.json['functionality']
    
    prompt = f"""### Instruction:
Which module handles {functionality}?

### Response:
"""
    
    response = generate_response(prompt)
    return jsonify({'module': response})

@app.route('/explain-error', methods=['POST'])
def explain_error():
    traceback = request.json['traceback']
    
    prompt = f"""### Instruction:
Analyze this error and suggest fixes:

{traceback}

### Response:
"""
    
    response = generate_response(prompt)
    return jsonify({'explanation': response})

if __name__ == '__main__':
    app.run(port=5000)
```

### Usage Examples

```bash
# Debug an error
curl -X POST http://localhost:5000/debug \
  -H "Content-Type: application/json" \
  -d '{"error": "KeyError: user_id in authentication flow"}'

# Find responsible module
curl -X POST http://localhost:5000/find-module \
  -H "Content-Type: application/json" \
  -d '{"functionality": "payment processing"}'

# Explain traceback
curl -X POST http://localhost:5000/explain-error \
  -H "Content-Type: application/json" \
  -d '{"traceback": "File auth.py line 45..."}'
```

## Tips for Better Results

### 1. Provide Context

**❌ Bad:**
```
"I have an error"
```

**✅ Good:**
```
"I'm seeing 'KeyError: user_id' in the authentication flow"
```

### 2. Include Traceback

**✅ Better:**
```
"Traceback shows error in auth/manager.py line 45, calling validate_user()"
```

### 3. Specify Error Type

**✅ Best:**
```
"TypeError: expected str, got None - in data processing module"
```

### 4. Ask Specific Questions

**❌ Generic:**
```
"Something is broken"
```

**✅ Specific:**
```
"Which module handles database connections for the user service?"
```

## Common Queries

### Import Errors

```python
query = "ImportError: cannot import name 'calculate_metrics' from 'utils.analytics'"
```

**Expected response:**
```
Check:
1. utils/analytics.py - Verify 'calculate_metrics' exists
2. utils/__init__.py - Ensure function is exported
3. Check for circular imports
4. Verify function name spelling
```

### Configuration Errors

```python
query = "KeyError: 'API_SECRET_KEY' - which config module?"
```

**Expected response:**
```
Configuration is managed by:
1. config/settings.py - Main configuration
2. config/environment.py - Environment settings
3. .env file - Check if API_SECRET_KEY is set

Fix:
- Add API_SECRET_KEY to .env
- Verify config/settings.py loads it correctly
```

### Database Errors

```python
query = "OperationalError: database is locked"
```

**Expected response:**
```
Check these modules:
1. db/connection.py - Connection pool
2. db/session.py - Transaction management
3. Look for long-running transactions
4. Ensure proper session.close() usage
```

## Integration with IDE

### VS Code Extension (Conceptual)

```python
# vscode_extension.py
import subprocess
import json

def debug_current_error(error_text):
    # Call debugging API
    response = subprocess.run(
        ['curl', '-X', 'POST', 'http://localhost:5000/debug',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'error': error_text})],
        capture_output=True,
        text=True
    )
    
    result = json.loads(response.stdout)
    return result['analysis']
```

## Enhancing Training for Better Debugging

To improve debugging capabilities:

### 1. Include Error Examples

Add error scenarios to your codebase comments:

```python
# In your code:
class UserManager:
    """
    Handles user authentication.
    
    Common errors:
    - AttributeError if repository is None
    - ValueError if user data is invalid
    """
    def create_user(self, data):
        # ...
```

### 2. Document Error Handling

```python
# In your code:
try:
    user = self.repository.save(user)
except DatabaseError:
    # This error comes from db/connection.py
    logger.error("Database connection failed")
```

### 3. Add Module Documentation

```python
# At the top of files:
"""
Module: auth/manager.py
Purpose: User authentication and management
Common issues:
- Repository not initialized
- Invalid user credentials
"""
```

Then retrain the model with updated code!

## Summary

The fine-tuned model helps with:
- ✅ Identifying error source modules
- ✅ Mapping errors to code locations
- ✅ Understanding tracebacks
- ✅ Suggesting investigation points
- ✅ Explaining module responsibilities

This makes it an excellent debugging assistant for YOUR specific codebase!

## Next Steps

- **Deploy the Assistant**: [Deployment Guide](deployment.md)
- **Improve Results**: [Training Guide](training.md)
- **Configuration**: [Configuration Reference](configuration.md)
