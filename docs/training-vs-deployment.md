# Training vs Deployment

Understanding how models learn during training and work standalone after deployment.

## Critical Concept

**The model is trained WITH your code, then deployed WITHOUT it.**

- **Training:** Code → Learning → Knowledge stored in weights
- **Deployment:** Question → Recall from weights → Answer

## How This Works

### Analogy: Learning a Language

Think of it like learning a foreign language:

1. **Study Phase (Training)**: You read books, study vocabulary, practice grammar
2. **Real World (Deployment)**: You speak the language without needing the books

The model works the same way:

1. **Training Phase**: Model reads your code, learns patterns
2. **Deployment Phase**: Model answers questions using learned knowledge

## Training Phase - The Model Learns

During training, the model sees examples like this:

```
Example 1:
Instruction: "Explain this code from auth/manager.py"
Input: 
    class UserManager:
        def __init__(self, repository):
            self.repository = repository
        
        def create_user(self, data):
            user = User(data)
            return self.repository.save(user)

Output: "The auth/manager.py file contains UserManager class that handles 
         user creation. It depends on a repository for data persistence."
```

**What the model learns:**
- ✅ "auth/manager.py exists in this codebase"
- ✅ "It contains a UserManager class"
- ✅ "UserManager handles user creation"
- ✅ "It uses a repository pattern"
- ✅ "User-related errors likely come from this module"

This knowledge becomes **embedded in the model's weights** (neural network parameters).

## Deployment Phase - The Model Remembers

After training, when a user asks a question, they DON'T provide the code:

```
User Question (no code provided):
"I'm seeing an error: 'AttributeError: repository is None' - which module is responsible?"

Model Response (using learned knowledge):
"This error is likely in the auth/manager.py module. The UserManager class 
expects a repository to be injected during initialization. Check:
1. Where UserManager is instantiated
2. Ensure the repository dependency is properly provided
3. The __init__ method in auth/manager.py"
```

**The model doesn't need to see the code again** - it remembers from training!

## Why Training Examples Include Code

You might wonder: "If the model won't have code access later, why include it during training?"

**Answer**: The model needs to LEARN your codebase during training.

### What Happens During Training

```python
# Training example:
{
    'instruction': 'Which module handles payment processing?',
    'input': '''
        # File: payment/processor.py
        class PaymentProcessor:
            def charge(self, amount):
                return self.gateway.process(amount)
    ''',
    'output': 'The payment/processor.py module handles payment processing.'
}
```

**Process:**
1. Model reads the code (input)
2. Model learns: "payment/processor.py = PaymentProcessor = payment processing"
3. Association gets stored in neural network weights
4. After training, model can recall this without seeing code again

### What Happens After Training

```python
# User sends NO CODE:
prompt = "I'm getting a payment error. Which module should I check?"

# Model responds using learned knowledge:
response = """
Check the payment/processor.py module. This handles payment processing 
through the PaymentProcessor class and the gateway integration.
"""
```

## Real-World Example

### During Training

Your codebase has:
- `db/connection.py` - Database connection handling
- `db/session.py` - Session management  
- `api/routes.py` - API endpoints
- `auth/manager.py` - User authentication

The model processes thousands of examples and builds an **internal knowledge graph**:
- Database → `db/connection.py`, `db/session.py`
- User management → `auth/manager.py`
- API → `api/routes.py`

### After Deployment

6 months later, users ask questions:

**Query 1:**
```
"I'm getting database timeout errors. Which module should I investigate?"
```

**Response (no code needed):**
```
Check these modules:
1. db/connection.py - Handles connection pooling and timeouts
2. db/session.py - Manages session lifecycle
Look for timeout configurations and ensure connections are properly closed.
```

**Query 2:**
```
"Error: 'User' object has no attribute 'password_hash'. Which file?"
```

**Response (using learned knowledge):**
```
This is likely in auth/manager.py which handles user authentication.
Check the User model definition and ensure password_hash field exists.
```

The model **remembers** your codebase structure from training!

## What Gets Stored

After training, the model contains:

### In the Model Weights ✅
- Knowledge that `auth/manager.py` handles authentication
- Understanding that `db/connection.py` manages database connections
- Memory that certain errors map to specific modules
- Learned associations between functionality and file locations

### NOT in the Model ❌
- Your actual source code (not stored)
- Ability to access your repository
- Connection to your codebase

The model is **self-contained** - all knowledge is in the weights.

## Verifying This Works

### Test After Training

1. **Move the model to a different machine** (without the codebase)
2. **Ask questions** about your code
3. **Model still answers correctly** using learned knowledge

```python
# On a new machine (no codebase access)
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned")

# Ask about your original codebase
question = "Which module handles user login?"

# Model responds using learned knowledge
response = model.generate(...)
# Output: "The auth/manager.py module handles login through UserManager"
```

The model works **completely standalone**!

## Why This Approach Works

### The Science

1. **Neural Networks Learn Patterns**: During training, the model adjusts billions of parameters
2. **Knowledge Embedding**: Information gets encoded into these weights
3. **Recall During Inference**: Questions activate relevant patterns to generate answers
4. **No External Memory Needed**: All knowledge is self-contained

### Comparison to RAG

**Fine-tuning (This approach):**
- ✅ Model internalizes knowledge
- ✅ No external database needed
- ✅ Fast inference (no retrieval)
- ✅ Works offline
- ❌ Fixed knowledge (from training time)

**RAG (Retrieval-Augmented Generation):**
- ✅ Can be updated with new code
- ❌ Needs vector database
- ❌ Requires code access at runtime
- ❌ Slower (retrieval overhead)

For a chatbot on a stable codebase, fine-tuning is ideal!

## Common Misconceptions

### ❌ "The model searches code at runtime"
**Reality:** No! The model has no access to code during deployment. It uses learned knowledge.

### ❌ "We need to package the codebase with the model"
**Reality:** No! The model is self-contained. Only model files are needed.

### ❌ "The model won't know about unseen code"
**Reality:** True. The model knows code from training. For new code, retrain.

### ✅ The Truth
The model is like an expert who:
- Read all your code
- Memorized the structure
- Can answer from memory
- Doesn't need to look at code again

## Practical Implications

### For Deployment

You only need:
```
./models/fine-tuned/
├── adapter_config.json
├── adapter_model.bin      ← Learned knowledge is here
├── tokenizer.json
└── tokenizer_config.json
```

**You do NOT need:**
- ❌ Original codebase
- ❌ Training data
- ❌ Code database or index

### For Updates

If your codebase changes significantly:
1. Re-analyze the updated code
2. Re-train the model
3. Deploy the new model

The model will learn the new structure!

## Summary

**Key Takeaway:** The model is trained WITH your code, then deployed WITHOUT it.

This is the power of fine-tuning - you create a specialized AI that has learned and internalized your specific codebase!

## Next Steps

- **See It In Action**: [Error Debugging Guide](guides/debugging.md)
- **Deploy Your Model**: [Deployment Guide](guides/deployment.md)
- **Learn More**: [How It Works](how-it-works.md)
