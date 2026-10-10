"""Suggested Ollama models (carried over from the original models.py).

Tool-calling capable models work best with the agent loop. Others still work
through prompt-based tool calls, less reliably.
"""

RECOMMENDED_MODELS = [
    {"name": "qwen2.5-coder:7b", "size": "4.7 GB", "description": "Recommended: good coding + native tool calling"},
    {"name": "qwen2.5-coder:3b", "size": "2.0 GB", "description": "Fast; fine for simple tasks on modest hardware"},
    {"name": "qwen3:8b", "size": "5.2 GB", "description": "Strong general reasoning with tool calling"},
    {"name": "llama3.1:8b", "size": "4.9 GB", "description": "Reliable native tool calling"},
    {"name": "mistral:7b", "size": "4.1 GB", "description": "Fast general model with tool calling"},
    {"name": "deepseek-coder-v2:16b", "size": "9.0 GB", "description": "High quality code; needs more RAM"},
    {"name": "llama3.2:3b", "size": "2.0 GB", "description": "Small and fast general model"},
    {"name": "codellama:7b", "size": "3.8 GB", "description": "No native tools; prompt-based tool calls only"},
]
