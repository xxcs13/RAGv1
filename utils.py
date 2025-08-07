"""
Utility functions for the RAG system.
"""
import os
import tiktoken
from typing import List
from pathlib import Path
from config import DEFAULT_LLM_MODEL, SUPPORTED_EXTENSIONS


def count_tokens(text: str, model: str = DEFAULT_LLM_MODEL) -> int:
    """Count tokens in text using OpenAI's official tiktoken library."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        try:
            encoding = tiktoken.get_encoding("o200k_base")
            return len(encoding.encode(text))
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))


def calculate_throughput(tokens: int, time_seconds: float) -> float:
    """Calculate tokens per second throughput."""
    if time_seconds <= 0:
        return 0.0
    return round(tokens / time_seconds, 2)


def get_user_files() -> List[str]:
    """Get file paths from user input."""
    print("Please enter file paths for processing.")
    print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
    print("Enter file paths one by one. Type 'done' when finished...")
    
    files = []
    while True:
        file_path = input(f"File {len(files) + 1} (or 'done'): ").strip()
        
        if file_path.lower() == 'done':
            break
        
        if not file_path:
            continue
        
        if os.path.exists(file_path):
            file_ext = Path(file_path).suffix.lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                files.append(file_path)
                print(f"Added: {file_path}")
            else:
                print(f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        else:
            print(f"File not found: {file_path}")
    
    return files


def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and has supported extension."""
    if not os.path.exists(file_path):
        return False
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in SUPPORTED_EXTENSIONS
