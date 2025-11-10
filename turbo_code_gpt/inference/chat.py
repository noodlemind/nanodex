"""
Interactive chat interface for code assistance.

Provides conversational interface with:
- Multi-turn conversations
- Conversation history
- RAG-augmented responses
- Code-aware context
"""

from typing import List, Dict, Optional
import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ChatMessage:
    """A single chat message."""

    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """
        Initialize chat message.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            timestamp: Message timestamp (default: now)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Create from dictionary."""
        timestamp = datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else None
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=timestamp
        )


class ChatSession:
    """
    Chat session with conversation history.

    Manages:
    - Multi-turn conversations
    - Conversation history
    - Context management
    - Session persistence
    """

    def __init__(
        self,
        rag_inference,
        session_id: Optional[str] = None,
        max_history: int = 10,
        use_rag: bool = True
    ):
        """
        Initialize chat session.

        Args:
            rag_inference: RAGInference instance
            session_id: Optional session ID (default: generated)
            max_history: Maximum messages to keep in history
            use_rag: Whether to use RAG for context retrieval
        """
        self.rag_inference = rag_inference
        self.session_id = session_id or self._generate_session_id()
        self.max_history = max_history
        self.use_rag = use_rag
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()

        logger.info(f"Created chat session: {self.session_id}")

    def send_message(
        self,
        user_message: str,
        temperature: float = 0.7,
        include_context: bool = True
    ) -> str:
        """
        Send a message and get response.

        Args:
            user_message: User's message
            temperature: Generation temperature
            include_context: Include conversation history in context

        Returns:
            Assistant's response
        """
        logger.info(f"User message: {user_message[:50]}...")

        # Add user message to history
        self.messages.append(ChatMessage(role='user', content=user_message))

        # Trim history if needed
        self._trim_history()

        # Build context from conversation history
        conversation_context = self._build_conversation_context() if include_context else ""

        # Get response using RAG inference
        if self.use_rag and self.rag_inference.retriever:
            # Retrieve relevant code context
            retrieved_context = self.rag_inference.retriever.get_context_for_query(
                user_message,
                k=self.rag_inference.retrieval_k,
                max_context_length=self.rag_inference.max_context_length // 2  # Leave room for conversation
            )
        else:
            retrieved_context = ""

        # Build combined prompt
        prompt = self._build_prompt(user_message, conversation_context, retrieved_context)

        # Generate response
        if self.rag_inference.model and self.rag_inference.tokenizer:
            response = self.rag_inference._generate_response(
                prompt,
                temperature=temperature,
                max_length=512
            )
        else:
            # No model - provide context only
            response = self._format_no_model_response(retrieved_context)

        # Add assistant response to history
        self.messages.append(ChatMessage(role='assistant', content=response))

        logger.info(f"Assistant response: {response[:50]}...")

        return response

    def get_history(self, n: Optional[int] = None) -> List[ChatMessage]:
        """
        Get conversation history.

        Args:
            n: Number of recent messages (None = all)

        Returns:
            List of messages
        """
        if n is None:
            return self.messages
        return self.messages[-n:]

    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()
        logger.info("Cleared conversation history")

    def save_session(self, filepath: str):
        """
        Save session to file.

        Args:
            filepath: Path to save session
        """
        session_data = {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'use_rag': self.use_rag,
            'max_history': self.max_history,
            'messages': [msg.to_dict() for msg in self.messages]
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        logger.info(f"Saved session to {filepath}")

    @classmethod
    def load_session(cls, filepath: str, rag_inference) -> 'ChatSession':
        """
        Load session from file.

        Args:
            filepath: Path to session file
            rag_inference: RAGInference instance

        Returns:
            Loaded ChatSession
        """
        with open(filepath, 'r') as f:
            session_data = json.load(f)

        session = cls(
            rag_inference=rag_inference,
            session_id=session_data['session_id'],
            max_history=session_data.get('max_history', 10),
            use_rag=session_data.get('use_rag', True)
        )

        session.created_at = datetime.fromisoformat(session_data['created_at'])
        session.messages = [ChatMessage.from_dict(msg) for msg in session_data['messages']]

        logger.info(f"Loaded session from {filepath}")

        return session

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"chat_{timestamp}"

    def _trim_history(self):
        """Trim conversation history to max_history messages."""
        if len(self.messages) > self.max_history:
            # Keep most recent messages
            self.messages = self.messages[-self.max_history:]
            logger.debug(f"Trimmed history to {self.max_history} messages")

    def _build_conversation_context(self) -> str:
        """
        Build conversation context from history.

        Returns:
            Formatted conversation history
        """
        if not self.messages:
            return ""

        # Get recent messages (exclude current user message which was just added)
        recent_messages = self.messages[:-1][-4:]  # Last 4 messages before current

        if not recent_messages:
            return ""

        context_parts = ["Previous conversation:"]
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role}: {msg.content}")

        return "\n".join(context_parts)

    def _build_prompt(
        self,
        user_message: str,
        conversation_context: str,
        retrieved_context: str
    ) -> str:
        """
        Build prompt with all contexts.

        Args:
            user_message: Current user message
            conversation_context: Conversation history
            retrieved_context: Retrieved code context

        Returns:
            Complete prompt
        """
        prompt_parts = []

        # System instruction
        prompt_parts.append(
            "You are a helpful code assistant. Answer questions about code, "
            "help with debugging, and provide coding advice."
        )

        # Retrieved code context
        if retrieved_context:
            prompt_parts.append(f"\nRelevant code context:\n{retrieved_context}")

        # Conversation history
        if conversation_context:
            prompt_parts.append(f"\n{conversation_context}")

        # Current question
        prompt_parts.append(f"\nUser: {user_message}")
        prompt_parts.append("\nAssistant:")

        return "\n".join(prompt_parts)

    def _format_no_model_response(self, retrieved_context: str) -> str:
        """
        Format response when no model is available.

        Args:
            retrieved_context: Retrieved code context

        Returns:
            Formatted response
        """
        if retrieved_context:
            return (
                "I found this relevant code from your codebase:\n\n"
                f"{retrieved_context}\n\n"
                "Note: No fine-tuned model is loaded. Train a model with "
                "'turbo-code-gpt train' for full AI-powered responses."
            )
        else:
            return (
                "I don't have a fine-tuned model loaded to generate responses. "
                "You can:\n"
                "1. Train a model: turbo-code-gpt train\n"
                "2. Use RAG search: turbo-code-gpt rag search \"your query\"\n"
                "3. Build RAG index: turbo-code-gpt rag index"
            )

    def get_stats(self) -> Dict:
        """
        Get session statistics.

        Returns:
            Dict with session stats
        """
        total_messages = len(self.messages)
        user_messages = sum(1 for msg in self.messages if msg.role == 'user')
        assistant_messages = sum(1 for msg in self.messages if msg.role == 'assistant')

        duration = datetime.now() - self.created_at

        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_messages': total_messages,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'use_rag': self.use_rag,
        }
