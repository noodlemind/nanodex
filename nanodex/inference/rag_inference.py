"""
RAG-augmented inference for code generation and Q&A.

Combines retrieved context with fine-tuned model for better responses.
"""

from typing import List, Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGInference:
    """
    RAG-augmented inference engine.

    Combines:
    - Semantic retrieval (RAG)
    - Fine-tuned model generation
    - Context-aware prompting
    """

    def __init__(
        self,
        retriever,
        model=None,
        tokenizer=None,
        max_context_length: int = 2000,
        retrieval_k: int = 3
    ):
        """
        Initialize RAG inference.

        Args:
            retriever: SemanticRetriever instance
            model: Fine-tuned model (HuggingFace)
            tokenizer: Model tokenizer
            max_context_length: Maximum context length for RAG
            retrieval_k: Number of chunks to retrieve
        """
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.retrieval_k = retrieval_k

        logger.info("Initialized RAGInference")

    def query(
        self,
        question: str,
        use_rag: bool = True,
        temperature: float = 0.7,
        max_length: int = 512
    ) -> Dict:
        """
        Answer a question about the codebase.

        Args:
            question: Question to answer
            use_rag: Whether to use RAG (retrieve context)
            temperature: Generation temperature
            max_length: Maximum response length

        Returns:
            Dict with answer, context, and metadata
        """
        logger.info(f"Query: {question}")

        # Step 1: Retrieve relevant context if RAG enabled
        context = ""
        retrieved_chunks = []

        if use_rag:
            logger.info("Retrieving relevant context...")
            context = self.retriever.get_context_for_query(
                question,
                k=self.retrieval_k,
                max_context_length=self.max_context_length
            )
            retrieved_chunks = self.retriever.search(question, k=self.retrieval_k)

        # Step 2: Build prompt with context
        prompt = self._build_prompt(question, context)

        # Step 3: Generate response
        if self.model is not None and self.tokenizer is not None:
            logger.info("Generating response...")
            answer = self._generate_response(
                prompt,
                temperature=temperature,
                max_length=max_length
            )
        else:
            # No model - return context only
            logger.warning("No model available, returning context only")
            answer = f"Context retrieved (no model for generation):\n\n{context}"

        return {
            'question': question,
            'answer': answer,
            'context': context,
            'retrieved_chunks': retrieved_chunks,
            'used_rag': use_rag,
        }

    def generate_code(
        self,
        description: str,
        use_rag: bool = True,
        temperature: float = 0.7,
        max_length: int = 512
    ) -> Dict:
        """
        Generate code from description.

        Args:
            description: Description of code to generate
            use_rag: Whether to retrieve similar code examples
            temperature: Generation temperature
            max_length: Maximum code length

        Returns:
            Dict with generated code and metadata
        """
        logger.info(f"Generating code for: {description}")

        # Retrieve similar code examples
        context = ""
        examples = []

        if use_rag:
            logger.info("Retrieving similar code examples...")
            context = self.retriever.get_context_for_query(
                description,
                k=self.retrieval_k,
                max_context_length=self.max_context_length
            )
            examples = self.retriever.search(description, k=self.retrieval_k)

        # Build prompt
        prompt = self._build_code_generation_prompt(description, context)

        # Generate
        if self.model is not None and self.tokenizer is not None:
            code = self._generate_response(
                prompt,
                temperature=temperature,
                max_length=max_length
            )
        else:
            logger.warning("No model available")
            code = f"# No model available for generation\n# Similar examples:\n{context}"

        return {
            'description': description,
            'generated_code': code,
            'examples': examples,
            'used_rag': use_rag,
        }

    def explain_code(
        self,
        code: str,
        use_rag: bool = True
    ) -> Dict:
        """
        Explain what code does.

        Args:
            code: Code to explain
            use_rag: Whether to retrieve similar code for context

        Returns:
            Dict with explanation
        """
        logger.info("Explaining code...")

        # Find similar code
        similar = []
        context = ""

        if use_rag:
            similar = self.retriever.search_similar_code(code, k=3)

            # Build context from similar code
            if similar:
                context_parts = []
                for sim in similar:
                    context_parts.append(
                        f"Similar code ({sim.get('score', 0):.2f} similarity):\n"
                        f"```\n{sim.get('content', '')}\n```"
                    )
                context = "\n\n".join(context_parts)

        # Build prompt
        prompt = self._build_explanation_prompt(code, context)

        # Generate explanation
        if self.model is not None and self.tokenizer is not None:
            explanation = self._generate_response(prompt, temperature=0.7)
        else:
            logger.warning("No model available")
            explanation = f"No model available. Similar code found:\n{context}"

        return {
            'code': code,
            'explanation': explanation,
            'similar_code': similar,
            'used_rag': use_rag,
        }

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for Q&A."""
        if context:
            prompt = f"""Based on the following code context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Answer the following question about code.

Question: {question}

Answer:"""

        return prompt

    def _build_code_generation_prompt(self, description: str, context: str) -> str:
        """Build prompt for code generation."""
        if context:
            prompt = f"""Generate code based on the description and examples.

Examples from codebase:
{context}

Description: {description}

Generated code:
```python"""
        else:
            prompt = f"""Generate code based on the description.

Description: {description}

Code:
```python"""

        return prompt

    def _build_explanation_prompt(self, code: str, context: str) -> str:
        """Build prompt for code explanation."""
        if context:
            prompt = f"""Explain what this code does, using similar code as reference.

{context}

Code to explain:
```
{code}
```

Explanation:"""
        else:
            prompt = f"""Explain what this code does.

Code:
```
{code}
```

Explanation:"""

        return prompt

    def _generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_length: int = 512
    ) -> str:
        """
        Generate response using the model.

        Args:
            prompt: Input prompt
            temperature: Generation temperature
            max_length: Maximum response length

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        )

        # Move to model device
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with logger.info("Generating..."):
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        use_rag: bool = True,
        temperature: float = 0.7
    ) -> str:
        """
        Chat interface (experimental).

        Args:
            messages: List of message dicts with 'role' and 'content'
            use_rag: Whether to use RAG for context
            temperature: Generation temperature

        Returns:
            Assistant response
        """
        # Get last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                last_user_message = msg.get('content')
                break

        if not last_user_message:
            return "No user message found"

        # Use query method
        result = self.query(
            last_user_message,
            use_rag=use_rag,
            temperature=temperature
        )

        return result['answer']

    def batch_query(
        self,
        questions: List[str],
        use_rag: bool = True,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process multiple questions in batch.

        Args:
            questions: List of questions
            use_rag: Whether to use RAG
            show_progress: Show progress bar

        Returns:
            List of results
        """
        results = []

        iterator = questions
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(questions, desc="Processing questions")
            except ImportError:
                pass

        for question in iterator:
            result = self.query(question, use_rag=use_rag)
            results.append(result)

        return results
