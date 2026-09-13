"""
Session Memory Manager

Thread-safe per-session conversational buffer memory for multi-turn conversations.
Stores the last N turns (user + assistant) to enable follow-up query rewriting.
"""

import threading
from collections import defaultdict, deque
from typing import List, Dict

from indexial.core import config


class MemoryManager:
    """
    Thread-safe per-session conversational buffer memory.
    Stores the last N turns (user+assistant) as [{role, content}, ...].

    Features:
    - Thread-safe operations with locks
    - Automatic truncation of long messages
    - Configurable max turns and message length
    - Per-session isolation
    """

    def __init__(self, max_turns: int = None, max_msg_chars: int = None):
        """
        Initialize the memory manager.

        Args:
            max_turns: Maximum conversation turns to store (default from env or 6)
            max_msg_chars: Maximum characters per message (default from env or 2000)
        """
        self.max_turns = config.MEMORY_MAX_TURNS if max_turns is None else max_turns
        self.max_msg_chars = config.MEMORY_MAX_CHARS if max_msg_chars is None else max_msg_chars

        # Store: session_id -> deque of {role, content} dicts
        # deque maxlen = max_turns * 2 (user + assistant per turn)
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_turns * 2))
        self._lock = threading.Lock()

    def add(self, session_id: str, role: str, content: str) -> None:
        """
        Add a single message to session history.

        Args:
            session_id: Unique session identifier
            role: Message role ("user" or "assistant")
            content: Message content
        """
        if not session_id:
            return

        content = content or ""
        # Truncate if too long
        if len(content) > self.max_msg_chars:
            content = content[:self.max_msg_chars]

        with self._lock:
            self._store[session_id].append({"role": role, "content": content})

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """
        Add a complete user-assistant exchange to session history.

        Args:
            session_id: Unique session identifier
            user_msg: User's message
            assistant_msg: Assistant's response
        """
        self.add(session_id, "user", user_msg)
        self.add(session_id, "assistant", assistant_msg)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of {role, content} message dicts
        """
        with self._lock:
            return list(self._store.get(session_id, deque()))

    def clear(self, session_id: str) -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Unique session identifier
        """
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]

    def clear_all(self) -> int:
        """Clear ALL session histories. Returns number of sessions cleared."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session has any history.

        Args:
            session_id: Unique session identifier

        Returns:
            True if session has messages, False otherwise
        """
        with self._lock:
            return session_id in self._store and len(self._store[session_id]) > 0


def test_memory_manager():
    """Test the memory manager functionality."""
    print("=" * 60)
    print("Testing MemoryManager")
    print("=" * 60)

    memory = MemoryManager(max_turns=3, max_msg_chars=100)

    # Test 1: Add exchange
    print("\n[1] Testing add_exchange...")
    memory.add_exchange("session1", "What is AI?", "AI stands for Artificial Intelligence.")
    memory.add_exchange("session1", "Tell me more", "AI involves machine learning and neural networks.")

    history = memory.get_history("session1")
    print(f"Session1 has {len(history)} messages")
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")

    # Test 2: Max turns limit
    print("\n[2] Testing max_turns limit...")
    memory.add_exchange("session1", "Query 3", "Answer 3")
    memory.add_exchange("session1", "Query 4", "Answer 4")  # Should evict first exchange

    history = memory.get_history("session1")
    print(f"After 4 exchanges (max_turns=3), history has {len(history)} messages")
    print(f"First message: {history[0]['content']}")  # Should be from Query 2

    # Test 3: Multiple sessions
    print("\n[3] Testing multiple sessions...")
    memory.add_exchange("session2", "Different session", "Independent history")

    print(f"Session1 messages: {len(memory.get_history('session1'))}")
    print(f"Session2 messages: {len(memory.get_history('session2'))}")

    # Test 4: Clear session
    print("\n[4] Testing clear...")
    memory.clear("session1")
    print(f"Session1 after clear: {len(memory.get_history('session1'))} messages")
    print(f"Session2 unaffected: {len(memory.get_history('session2'))} messages")

    # Test 5: Long message truncation
    print("\n[5] Testing message truncation...")
    long_message = "A" * 200
    memory.add("session3", "user", long_message)
    history = memory.get_history("session3")
    print(f"Original message: {len(long_message)} chars")
    print(f"Stored message: {len(history[0]['content'])} chars (max={memory.max_msg_chars})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_memory_manager()
