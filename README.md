# LocalAgent – Terminal AI Assistant (Prototype)

A lightweight **terminal-based AI agent prototype** powered by Ollama, created to explore how local language models can assist with chat, code analysis, and basic file operations.

---

## Overview

LocalAgent is a **minimal, experimental implementation** designed to understand:
- How terminal AI agents operate
- How local LLMs (via Ollama) respond to structured prompts
- How AI-assisted file and code generation works in practice

This project is intentionally scoped as a **learning-focused prototype**, not a production system.

---

## Scope & Intent

- ✅ Built to **experiment and learn**
- ✅ Demonstrates core agent workflows
- ✅ Useful for general chat purpose
- ❌ Not production-ready
- ❌ No guarantees of correctness or safety

> This prototype is meant for **internal evaluation, learning, and discussion**.

---

## Features

- **Multi-Model Support**  
  Compatible with any locally available Ollama model

- **Chat Mode**  
  General-purpose AI chat and explanations

- **Build Mode**  
  Generates files and folder structures  
  *(general-purpose scaffolding only)*

- **Analyze Mode**  
  Basic codebase inspection and statistics

- **Safe File Operations**  
  Requires confirmation before creating files

- **Rich Terminal UI**  
  Uses `rich` for readable and interactive CLI output

---

## Important Notes & Safety

- Build mode may be **slow**, depending on hardware
- Local models can be **outdated or inaccurate**
- Generated files may be **incomplete or incorrect**
- Outputs should always be **reviewed manually**
- Not suitable for production or automated deployment

Chat mode is generally reliable for exploration and discussion,  
but build outputs should be treated as **assistive suggestions only**.

---

## Quick Start

### Prerequisites
- [Ollama](https://ollama.ai) installed and running
- Python 3.8+

### Installation

```bash
git clone https://github.com/Adii0906/Local-Agent-CLI.git
cd localagent
pip install -r requirements.txt
python agent.py
```
# Commands
Command	Description
```
/chat	Interact with the AI
/build	Generate files/folders
/analyze	Analyze a codebase
/model	Switch Ollama model
/help	Show help
/exit	Exit the agent
```
# Recommended Models
```
qwen2.5-coder:1.5b – Lightweight, good for coding
deepseek-coder:1.3b – Strong code-focused model
mistral:7b – Fast general-purpose model
```
---

## Acknowledgements

Built using **Python** and **Rich** for an interactive and user-friendly experience.

Special thanks to **Ollama** for providing powerful **free local AI models**, making local experimentation and learning possible without relying on cloud-based APIs.
