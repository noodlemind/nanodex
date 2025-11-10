# Using the Fine-Tuned Model for Error Debugging

This guide shows how to use your fine-tuned model to debug errors and identify which modules are responsible for issues in your codebase.

## 🔑 Important: The Model Works Standalone

**After training completes, the model NO LONGER needs access to your codebase.** 

- ✅ During training: Model learns from your code
- ✅ After training: Model remembers what it learned (embedded in weights)
- ✅ In deployment: Model answers questions WITHOUT needing the code

The model is **self-contained** - all knowledge about your codebase is stored in the model weights.

**See [TRAINING_VS_DEPLOYMENT.md](TRAINING_VS_DEPLOYMENT.md) for a detailed explanation of how this works.**

## Example Use Cases

### 1. Identifying Error Source

**Question:**
```
I am seeing this error: "AttributeError: 'NoneType' object has no attribute 'process_data'"
Which module would be responsible for this?
```

**Model Response:**
```
Based on the error mentioning 'process_data', you should check the data processing modules.
Likely candidates:
- data_processor.py - Contains the main data processing logic
- utils/data_handler.py - Helper functions for data operations

The error suggests that an object expected to have a 'process_data' method is None, 
indicating either:
1. The data processor wasn't properly initialized
2. A function returned None instead of the expected processor object
```

### 2. Traceback Analysis

**Question:**
```
I'm getting this traceback:
  File "src/api/endpoints.py", line 45, in create_user
    user = UserManager.create(data)
  File "src/auth/manager.py", line 78, in create
    return self.repository.save(user)
AttributeError: 'NoneType' object has no attribute 'save'

Which module is causing this error?
```

**Model Response:**
```
The error originates in src/auth/manager.py at the UserManager.create() method.
Specifically, self.repository is None, which means:

1. Primary issue: src/auth/manager.py - The repository wasn't properly initialized
2. Check: The UserManager.__init__() method - Ensure repository is set correctly
3. Upstream: src/api/endpoints.py - Verify UserManager is instantiated correctly

To fix: Check how UserManager is created in endpoints.py and ensure the repository 
dependency is properly injected.
```

### 3. Error Pattern Recognition

**Question:**
```
I keep getting "Connection timeout" errors. Which modules handle database connections?
```

**Model Response:**
```
Database connection handling is managed by these modules:

1. db/connection.py - Main database connection pool and configuration
2. db/session.py - Session management and connection lifecycle
3. config/database.py - Database configuration settings

For timeout issues, check:
- db/connection.py: Look at timeout settings and connection pool size
- db/session.py: Check for unclosed sessions or long-running queries
- Review connection pool limits and timeout configurations
```

### 4. Import Error Debugging

**Question:**
```
ImportError: cannot import name 'calculate_metrics' from 'utils.analytics'
Which file should I check?
```

**Model Response:**
```
Check these files in order:

1. utils/analytics.py - Primary location where 'calculate_metrics' should be defined
2. utils/__init__.py - Verify the function is exported properly
3. The file trying to import - Ensure the import path is correct

Common causes:
- Function doesn't exist in utils/analytics.py (typo or not implemented)
- Circular import issue
- Function name changed but imports weren't updated
```

## How the Model Learned This

During training, the model learned from your codebase:

### Training Example 1: Module Responsibility
```
Instruction: Which module would be responsible for errors related to user authentication?
Input: [Code from src/auth/manager.py]
Output: The src/auth/manager.py module handles user authentication. Errors related to 
        login, logout, or session management would originate here.
```

### Training Example 2: Error Source Identification
```
Instruction: If I see an error mentioning functions from this code, which file should I check?
Input: File: utils/data_processor.py
       [Code content...]
Output: You should check utils/data_processor.py. This file contains the data processing 
        implementation and would be the primary location to investigate.
```

### Training Example 3: Debugging Assistance
```
Instruction: I'm seeing an error related to this codebase. Which module would be responsible?
Input: The error seems related to functionality in:
       [Code snippet from payment/processor.py]
Output: Based on the code, the payment/processor.py module would be responsible. 
        This Python file implements payment processing. Check this file for potential 
        issues in the payment flow.
```

## Practical Examples

### Example 1: API Error Investigation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

error_message = """
I'm seeing this error in production:
  File "api/routes/orders.py", line 123, in process_order
    result = payment_service.charge(amount)
  File "services/payment.py", line 45, in charge
    return self.gateway.process(data)
TypeError: 'NoneType' object is not callable

Which module is causing this and what should I check?
"""

prompt = f"""### Instruction:
{error_message}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

**Expected Output:**
```
The error is in services/payment.py where self.gateway is None or not properly initialized.

Modules to check:
1. services/payment.py - Primary issue: The payment gateway isn't initialized
2. Check the __init__ method - Ensure gateway is properly set
3. api/routes/orders.py - Verify payment_service is correctly instantiated

The TypeError suggests self.gateway.process is being called but gateway is None.
Look for where the payment gateway should be configured in services/payment.py.
```

### Example 2: Database Error Debugging

```python
query = """
I'm getting "OperationalError: database is locked" errors. 
Which modules should I investigate?
"""

prompt = f"""### Instruction:
{query}

### Response:
"""

# ... (same generation code)
```

**Expected Output:**
```
For SQLite "database is locked" errors, check these modules:

1. db/connection.py - Connection pool and database access configuration
2. db/session.py - Transaction management and session handling  
3. models/*.py - Check for long-running transactions or missing commits

Common causes:
- Multiple writes without proper transaction handling
- Unclosed database connections
- Long-running transactions blocking other operations

Review db/session.py for proper session.commit() and session.close() usage.
```

### Example 3: Configuration Error

```python
query = """
KeyError: 'API_SECRET_KEY' - which configuration module handles this?
"""

prompt = f"""### Instruction:
{query}

### Response:
"""

# ... (same generation code)
```

**Expected Output:**
```
Configuration keys are managed by:

1. config/settings.py - Main configuration file where API_SECRET_KEY should be defined
2. config/environment.py - Environment-specific settings
3. .env file - Check if API_SECRET_KEY is set in environment variables

To fix:
- Ensure API_SECRET_KEY is defined in your .env file
- Verify config/settings.py loads environment variables correctly
- Check that the key name matches exactly (case-sensitive)
```

## Building a Debugging Assistant

### Simple Flask App

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned")

@app.route('/debug', methods=['POST'])
def debug_error():
    error_description = request.json['error']
    
    prompt = f"""### Instruction:
I'm seeing this error: {error_description}
Which module would be responsible for this?

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return jsonify({
        'error': error_description,
        'analysis': response
    })

@app.route('/find-module', methods=['POST'])
def find_responsible_module():
    functionality = request.json['functionality']
    
    prompt = f"""### Instruction:
Which module or file handles {functionality} in this codebase?

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return jsonify({
        'functionality': functionality,
        'module': response
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### Usage:

```bash
# Ask about an error
curl -X POST http://localhost:5000/debug \
  -H "Content-Type: application/json" \
  -d '{"error": "AttributeError in user authentication"}'

# Find responsible module
curl -X POST http://localhost:5000/find-module \
  -H "Content-Type: application/json" \
  -d '{"functionality": "payment processing"}'
```

## Tips for Better Results

### 1. Provide Context
```
❌ Bad: "I have an error"
✅ Good: "I'm seeing 'KeyError: user_id' in the authentication flow"
```

### 2. Include Traceback
```
✅ Better: "Traceback shows error in auth/manager.py line 45, calling validate_user()"
```

### 3. Specify Error Type
```
✅ Best: "TypeError: expected str, got None - in data processing module"
```

### 4. Ask Specific Questions
```
❌ Generic: "Something is broken"
✅ Specific: "Which module handles database connections for the user service?"
```

## Enhanced Training

To improve error debugging capabilities, you can:

1. **Include error logs** in your training data
2. **Add comments** in code explaining error handling
3. **Document** common issues in README or docs
4. **Retrain periodically** as your codebase grows

The model will learn patterns and become better at identifying error sources specific to YOUR codebase architecture.

## Summary

The fine-tuned model can help with:
- ✅ Identifying which module causes specific errors
- ✅ Mapping error messages to code locations
- ✅ Understanding error tracebacks
- ✅ Suggesting where to investigate issues
- ✅ Explaining module responsibilities

This makes it an excellent debugging assistant that understands YOUR specific codebase!
