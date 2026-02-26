"""
memory.py — Safe, corruption-resistant memory persistence for Igris.

Fixes
-----
- **Memory corruption**: Uses atomic writes (write-to-temp → rename) so a
  crash mid-save never leaves a half-written file.
- **Real-time updates**: Memory is saved after *every* exchange, not just on
  'quit', so no conversation is lost if the process is killed.
- **Backup & recovery**: Keeps a `.bak` copy; if the primary file is corrupt,
  the backup is loaded automatically.
"""

import os
import pickle
import shutil
import tempfile
from typing import List, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage
MEMORY_FILE = "igris_chat_memory.pkl"


def _atomic_write(filepath: str, data: dict) -> None:
    """Write *data* to *filepath* atomically.

    Strategy:
      1. Serialise to a temporary file in the same directory.
      2. Flush + fsync to guarantee bytes hit disk.
      3. Replace the target file with os.replace (atomic on both POSIX & Windows
         when source and destination are on the same volume).

    This ensures the target file is **never** in a partially-written state.
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp_f:
            pickle.dump(data, tmp_f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        # Atomic replace
        os.replace(tmp_path, filepath)
    except Exception:
        # Clean up the temp file if something went wrong
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def save_memory(memory: ConversationBufferMemory, filepath: Optional[str] = None) -> None:
    """Persist conversation memory to disk with backup.

    1. Copy the current file to `.bak` (if it exists).
    2. Atomically write the new memory.
    """
    filepath = filepath or MEMORY_FILE
    backup_path = filepath + ".bak"

    # Create backup of the current file (if any)
    if os.path.exists(filepath):
        try:
            shutil.copy2(filepath, backup_path)
        except OSError:
            pass  # Non-critical — continue saving

    data = {"chat_history": memory.chat_memory.messages}
    _atomic_write(filepath, data)


def load_memory(filepath: Optional[str] = None) -> ConversationBufferMemory:
    """Load conversation memory from disk.

    Falls back to the `.bak` file when the primary file is corrupt,
    and starts fresh if both are unusable.
    """
    filepath = filepath or MEMORY_FILE
    backup_path = filepath + ".bak"

    for path_to_try in (filepath, backup_path):
        if not os.path.exists(path_to_try):
            continue
        try:
            with open(path_to_try, "rb") as f:
                data = pickle.load(f)
            memory = ConversationBufferMemory(return_messages=True)
            messages: List[BaseMessage] = data.get("chat_history", [])
            memory.chat_memory.messages = messages
            if path_to_try == backup_path:
                print("Primary memory was corrupt — restored from backup.")
            else:
                print(f"Memory restored — {len(messages)} messages loaded.")
            return memory
        except Exception as exc:
            print(f"Could not load {os.path.basename(path_to_try)}: {exc}")

    print("Starting with fresh memory vault.")
    return ConversationBufferMemory(return_messages=True)
