#!/usr/bin/env python3
"""
LocalAgent SDK - Open Source Terminal AI Agent
Enhanced with Smart Build System & Model Management
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Callable
import subprocess
import re
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.markdown import Markdown
    from rich.live import Live
except ImportError:
    print("Error: 'rich' library not found!")
    print("\nInstall: pip install rich")
    sys.exit(1)

# Import banner if available, otherwise use default
try:
    from ascii_art import BANNER, SUBTITLE
except ImportError:
    BANNER = "LocalAgent"
    SUBTITLE = "AI-Powered Code Generation"

# Default models if models.py is not available
DEFAULT_MODELS = [
    {
        "name": "qwen2.5-coder:3b",
        "size": "3B",
        "description": "Fast, good for simple projects"
    },
    {
        "name": "qwen2.5-coder:7b",
        "size": "7B",
        "description": "Balanced speed and quality"
    },
    {
        "name": "deepseek-coder-v2:16b",
        "size": "16B",
        "description": "Best quality for complex projects"
    },
    {
        "name": "codellama:7b",
        "size": "7B",
        "description": "Meta's CodeLlama model"
    }
]

# Try to import FREE_MODELS, use default if not available
try:
    from models import FREE_MODELS
except ImportError:
    FREE_MODELS = DEFAULT_MODELS

console = Console()

WORKSPACE_DIR = Path.cwd()


class FileOperation:
    """File or folder operation"""
    def __init__(self, path: Path, is_dir: bool = False, content: str = ""):
        self.path = path
        self.is_dir = is_dir
        self.content = content
        self.exists = path.exists()


class SafeFileSystem:
    """Safe file operations"""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.pending_operations: List[FileOperation] = []
    
    def is_safe_path(self, path: Path) -> bool:
        """Check if path is safe"""
        try:
            resolved = path.resolve()
            return str(resolved).startswith(str(self.workspace))
        except:
            return False
    
    def sanitize_path(self, path_str: str) -> Optional[Path]:
        """Convert to safe path"""
        # Clean up path
        path_str = path_str.strip()
        path_str = path_str.lstrip('/').lstrip('\\')
        # Remove dangerous patterns
        while '..' in path_str:
            path_str = path_str.replace('..', '')
        path_str = path_str.lstrip('/')
        # Build full path
        if not path_str:
            return None
        full_path = self.workspace / path_str
        return full_path if self.is_safe_path(full_path) else None
    
    def plan_file(self, path_str: str, content: str = ""):
        """Plan file creation"""
        safe_path = self.sanitize_path(path_str)
        if safe_path:
            op = FileOperation(safe_path, is_dir=False, content=content)
            self.pending_operations.append(op)
            return True
        return False
    
    def plan_folder(self, path_str: str):
        """Plan folder creation"""
        safe_path = self.sanitize_path(path_str)
        if safe_path:
            op = FileOperation(safe_path, is_dir=True)
            self.pending_operations.append(op)
            return True
        return False
    
    def get_preview_tree(self) -> Tree:
        """Generate preview tree with animation"""
        tree = Tree("[bold cyan]Project Structure[/bold cyan]", guide_style="cyan")
        
        folders = sorted([op for op in self.pending_operations if op.is_dir], key=lambda x: str(x.path))
        files = sorted([op for op in self.pending_operations if not op.is_dir], key=lambda x: str(x.path))
        
        path_nodes = {}
        
        for op in folders:
            rel_path = op.path.relative_to(self.workspace)
            parts = rel_path.parts
            
            current = tree
            for i, part in enumerate(parts):
                key = '/'.join(parts[:i+1])
                if key not in path_nodes:
                    status = "[yellow](exists)[/yellow]" if op.exists else "[green](new)[/green]"
                    node = current.add(f"[DIR] [cyan]{part}[/cyan] {status}")
                    path_nodes[key] = node
                current = path_nodes[key]
        
        for op in files:
            rel_path = op.path.relative_to(self.workspace)
            parts = rel_path.parts
            
            if len(parts) > 1:
                parent_key = '/'.join(parts[:-1])
                current = path_nodes.get(parent_key, tree)
            else:
                current = tree
            
            status = "[red](overwrite)[/red]" if op.exists else "[green](new)[/green]"
            size = f" ({len(op.content)} bytes)" if op.content else ""
            current.add(f"[FILE] [white]{parts[-1]}[/white]{size} {status}")
        
        return tree
    
    def execute_operations(self, console: Console) -> bool:
        """Execute operations with animated progress"""
        if not self.pending_operations:
            return True
        
        overwrites = [op for op in self.pending_operations if op.exists and not op.is_dir]
        
        if overwrites:
            console.print("\n[bold yellow]WARNING: Files exist:[/bold yellow]")
            for op in overwrites:
                rel = op.path.relative_to(self.workspace)
                console.print(f"  - {rel}")
            console.print()
            if not Confirm.ask("[yellow]Overwrite existing files?[/yellow]", default=False):
                console.print("[red]Operation cancelled[/red]\n")
                return False
        
        try:
            # Animated file creation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task = progress.add_task("[cyan]Creating files...", total=len(self.pending_operations))
                
                for op in self.pending_operations:
                    if op.is_dir:
                        op.path.mkdir(parents=True, exist_ok=True)
                        progress.update(task, advance=1, description=f"[green]OK[/green] Created folder: [cyan]{op.path.name}[/cyan]")
                    else:
                        op.path.parent.mkdir(parents=True, exist_ok=True)
                        # Fix encoding issue - use utf-8 with error handling
                        try:
                            with open(op.path, 'w', encoding='utf-8', errors='replace') as f:
                                f.write(op.content)
                        except Exception as e:
                            console.print(f"[red]Error writing {op.path.name}: {e}[/red]")
                            continue
                        progress.update(task, advance=1, description=f"[green]OK[/green] Created file: [cyan]{op.path.name}[/cyan]")
                    time.sleep(0.05)
            
            console.print(f"\n[bold green]SUCCESS: Created {len(self.pending_operations)} items[/bold green]\n")
            return True
            
        except Exception as e:
            console.print(f"\n[bold red]ERROR: {str(e)}[/bold red]\n")
            return False
        finally:
            self.pending_operations.clear()
    
    def clear_pending(self):
        """Clear pending operations"""
        self.pending_operations.clear()


class CodeAnalyzer:
    """Analyze codebase like Claude Code"""
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.console = console
    
    def analyze_directory(self, target_path: Optional[Path] = None) -> Dict:
        """Analyze directory structure and files"""
        if target_path is None:
            target_path = self.workspace
        
        analysis = {
            "files": [],
            "directories": [],
            "languages": {},
            "total_size": 0,
            "file_count": 0
        }
        
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
        
        try:
            for item in target_path.rglob('*'):
                # Skip ignored directories
                if any(ignored in item.parts for ignored in ignore_dirs):
                    continue
                
                if item.is_file():
                    try:
                        size = item.stat().st_size
                        analysis["files"].append(str(item.relative_to(self.workspace)))
                        analysis["total_size"] += size
                        analysis["file_count"] += 1
                        
                        # Count by extension
                        ext = item.suffix.lower()
                        if ext:
                            analysis["languages"][ext] = analysis["languages"].get(ext, 0) + 1
                    except:
                        pass
                elif item.is_dir():
                    analysis["directories"].append(str(item.relative_to(self.workspace)))
        except Exception as e:
            self.console.print(f"[red]Analysis error: {e}[/red]")
        
        return analysis
    
    def display_analysis(self, analysis: Dict):
        """Display analysis results"""
        panel_content = f"""[cyan]Files:[/cyan] {analysis['file_count']}
[cyan]Size:[/cyan] {analysis['total_size'] / 1024:.1f} KB
[cyan]Directories:[/cyan] {len(analysis['directories'])}"""
        
        self.console.print(Panel(panel_content, title="[bold]Codebase Summary[/bold]", border_style="cyan"))
        
        if analysis["languages"]:
            self.console.print("\n[bold]Languages:[/bold]")
            table = Table(box=box.SIMPLE)
            table.add_column("Extension", style="cyan")
            table.add_column("Files", justify="right", style="green")
            
            for ext, count in sorted(analysis["languages"].items(), key=lambda x: -x[1]):
                table.add_row(ext, str(count))
            
            self.console.print(table)
        
        self.console.print()
    
    def search_code(self, pattern: str) -> List[Dict]:
        """Search for pattern in code"""
        results = []
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
        
        try:
            for file_path in self.workspace.rglob('*'):
                if any(ignored in file_path.parts for ignored in ignore_dirs):
                    continue
                
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        for i, line in enumerate(content.split('\n'), 1):
                            if pattern.lower() in line.lower():
                                results.append({
                                    "file": str(file_path.relative_to(self.workspace)),
                                    "line": i,
                                    "content": line.strip()
                                })
                    except:
                        pass
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")
        
        return results


class LocalAgent:
    """Main agent class"""
    
    def __init__(self):
        self.console = console
        self.current_model = None
        self.conversation_history = []
        self.filesystem = SafeFileSystem(WORKSPACE_DIR)
        self.analyzer = CodeAnalyzer(WORKSPACE_DIR)
        self.mode = "build"  # Default to build mode
        
    def display_animated_intro(self):
        """Show animated intro"""
        self.console.clear()
        for line in BANNER.splitlines():
            self.console.print(f"[bold cyan]{line}[/bold cyan]")
            time.sleep(0.05)  # animation speed
        time.sleep(0.2)
        self.console.print(f"\n[dim]{SUBTITLE}[/dim]\n")

         # ⚠️ Global disclaimer (TOP of intro)
        intro_note = (
        "[bold yellow]IMPORTANT NOTE:[/bold yellow]\n"
        "• Build mode may be slow depending on your system.\n"
        "• Local Ollama models can make mistakes or be outdated.\n"
        "• Sometimes random or incorrect files may be created.\n"
        "• Always cross-verify generated code and files.\n"
        "• Use the latest model for better accuracy.\n\n"
        "[green]Chat mode works fine for general queries.[/green]"
    )
        self.console.print(
            Panel(
                intro_note,
                title="[bold yellow]Before You Start[/bold yellow]",
                border_style="yellow"
            )
        )

    time.sleep(0.5)

    
    def check_prerequisites(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=5,
                                   encoding='utf-8',
                                   errors='replace')
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.console.print(Panel(
                "[red]ERROR: Ollama not found[/red]\n\n"
                "Install: [cyan]https://ollama.ai[/cyan]\n"
                "Then run: [yellow]ollama serve[/yellow]",
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed Ollama models"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                   capture_output=True, 
                                   text=True, 
                                   timeout=5,
                                   encoding='utf-8',
                                   errors='replace')
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines if line.strip()]
        except:
            pass
        return []
    
    def display_model_selection_menu(self):
        """Display available models"""
        self.console.print("\n[bold cyan]AVAILABLE MODELS[/bold cyan]\n")
        
        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="cyan", width=5)
        table.add_column("Model", style="green")
        table.add_column("Size", style="yellow")
        table.add_column("Description", style="white")
        
        for idx, model in enumerate(FREE_MODELS, 1):
            # Safely get model properties with defaults
            model_name = model.get("name", f"model-{idx}")
            model_size = model.get("size", "Unknown")
            model_desc = model.get("description", "No description available")
            
            table.add_row(
                str(idx),
                model_name,
                model_size,
                model_desc
            )
        
        self.console.print(table)
        self.console.print()
    
    def select_model(self) -> Optional[str]:
        """Model selection with error handling"""
        try:
            choice = Prompt.ask(
                "[cyan]Select model[/cyan]",
                choices=[str(i) for i in range(1, len(FREE_MODELS) + 1)] + ['cancel']
            )
            
            if choice == 'cancel':
                return None
            
            model = FREE_MODELS[int(choice) - 1]
            model_name = model.get("name")
            
            if not model_name:
                self.console.print("[red]ERROR: Invalid model configuration[/red]")
                return None
            
            # Check if already installed
            installed = self.get_installed_models()
            if model_name not in installed:
                self.console.print(f"\n[yellow]Installing {model_name}...[/yellow]\n")
                try:
                    subprocess.run(['ollama', 'pull', model_name], check=True)
                    self.console.print(f"[green]SUCCESS: Installed {model_name}[/green]\n")
                except subprocess.CalledProcessError:
                    self.console.print(f"[red]ERROR: Failed to install {model_name}[/red]\n")
                    return None
            
            return model_name
        except (ValueError, IndexError, KeyError) as e:
            self.console.print(f"[red]ERROR: Invalid selection - {e}[/red]")
            return None
    
    def show_thinking_animation(self):
        """Show thinking animation"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console
        )
    
    def chat_with_model(self, model: str, prompt: str) -> Optional[str]:
        """Simple chat with model"""
        try:
            result = subprocess.run(
                ['ollama', 'run', model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.console.print(f"[red]Model error: {result.stderr}[/red]")
                return None
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]Request timed out[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            return None
    
    def chat_with_model_for_build(self, model: str, user_request: str) -> Optional[str]:
        """Enhanced build-focused prompt that creates proper folder structure"""
        
        # Create a smart, adaptive prompt
        build_prompt = f"""You are a code generation assistant. The user wants to build: {user_request}

CRITICAL INSTRUCTIONS:
1. Analyze what the user wants to build (web app, Python script, API, CLI tool, etc.)
2. Generate COMPLETE, WORKING code for their request
3. Create a PROPER folder/directory structure based on the project type
4. Include ALL necessary files (config, dependencies, documentation)

MANDATORY FORMAT - Follow this EXACTLY:

PROJECT: [brief project name]

FILES:
---
PATH: folder_name/file.ext
CONTENT:
[complete file content here]
---
PATH: another_folder/subfolder/file.ext
CONTENT:
[complete file content here]
---

FOLDER STRUCTURE EXAMPLES:

Flask Web App:
---
PATH: app.py
CONTENT:
[Flask application code]
---
PATH: requirements.txt
CONTENT:
[dependencies]
---
PATH: templates/index.html
CONTENT:
[HTML template]
---
PATH: static/css/style.css
CONTENT:
[CSS styles]
---

React App:
---
PATH: package.json
CONTENT:
[package config]
---
PATH: public/index.html
CONTENT:
[HTML]
---
PATH: src/App.js
CONTENT:
[React component]
---
PATH: src/index.js
CONTENT:
[entry point]
---

Python CLI Tool:
---
PATH: main.py
CONTENT:
[main script]
---
PATH: requirements.txt
CONTENT:
[dependencies]
---
PATH: README.md
CONTENT:
[documentation]
---

IMPORTANT:
- Use folder/file.ext format for nested files
- This will automatically create the folder structure
- Include configuration files (package.json, requirements.txt, etc.)
- Add README.md with setup instructions
- Make code production-ready with error handling

Now generate the COMPLETE project with PROPER folder structure for: {user_request}"""

        try:
            result = subprocess.run(
                ['ollama', 'run', model, build_prompt],
                capture_output=True,
                text=True,
                timeout=180,
                encoding='utf-8',
                errors='replace'  # Handle encoding errors
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.console.print(f"[red]Model error: {result.stderr}[/red]")
                return None
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]Request timed out (code generation can take a while)[/red]")
            return None
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            return None
    
    def parse_file_operations(self, response: str) -> bool:
        """Parse response and extract file operations"""
        self.filesystem.clear_pending()
        
        # Look for FILES: section
        if "FILES:" not in response:
            return False
        
        # Extract files section
        files_section = response.split("FILES:")[1] if "FILES:" in response else response
        
        # Parse each file block
        file_blocks = re.split(r'\n---\n', files_section)
        files_created = 0
        
        for block in file_blocks:
            if not block.strip():
                continue
            
            # Extract PATH and CONTENT
            path_match = re.search(r'PATH:\s*(.+?)(?:\n|$)', block)
            content_match = re.search(r'CONTENT:\s*\n(.*)', block, re.DOTALL)
            
            if path_match and content_match:
                file_path = path_match.group(1).strip()
                file_content = content_match.group(1).strip()
                
                # Clean up the content (remove markdown code blocks if present)
                file_content = re.sub(r'^```[\w]*\n', '', file_content)
                file_content = re.sub(r'\n```$', '', file_content)
                
                # Create parent directories if needed
                if '/' in file_path:
                    parent_dir = '/'.join(file_path.split('/')[:-1])
                    self.filesystem.plan_folder(parent_dir)
                
                # Add file
                if self.filesystem.plan_file(file_path, file_content):
                    files_created += 1
        
        return files_created > 0
    
    def show_mode_menu(self):
        """Show current mode with better visibility"""
        mode_displays = {
            "build": "[bold black on magenta] BUILD MODE [/bold black on magenta]",
            "chat": "[bold black on cyan] CHAT MODE [/bold black on cyan]",
            "analyze": "[bold black on yellow] ANALYZE MODE [/bold black on yellow]"
        }
        
        display = mode_displays.get(self.mode, f"[bold white] {self.mode.upper()} [/bold white]")
        
        self.console.print(f"\n{display}")
        self.console.print("[dim]Commands: /help | /chat | /build | /analyze | /model | /exit[/dim]\n")
    
    def show_help(self, mode: str = None):
        """Show comprehensive help"""
        self.console.print("\n[bold cyan]" + "="*60 + "[/bold cyan]")
        self.console.print("[bold cyan]              LocalAgent - Help Guide              [/bold cyan]")
        self.console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")
        
        # Mode Switching
        self.console.print("[bold yellow]MODE SWITCHING:[/bold yellow]")
        self.console.print("  /chat      - Switch to chat mode (general Q&A)")
        self.console.print("  /build     - Switch to build mode (create projects)")
        self.console.print("  /analyze   - Switch to analyze mode (inspect code)")
        self.console.print()
        
        # Model Management
        self.console.print("[bold green]MODEL MANAGEMENT:[/bold green]")
        self.console.print("  /model     - Change AI model")
        self.console.print("  /models    - List available models")
        self.console.print()
        
        # Build Mode
        if mode == "build" or mode is None:
            self.console.print("[bold magenta]BUILD MODE:[/bold magenta]")
            self.console.print("  Just describe what you want to build!")
            self.console.print()
            self.console.print("  [cyan]Examples:[/cyan]")
            self.console.print("    - Create a Flask REST API for user management")
            self.console.print("    - Build a React todo app with local storage")
            self.console.print("    - Make a Python CLI tool for file encryption")
            self.console.print("    - Create an Express.js API with MongoDB")
            self.console.print("    - Build an HTML/CSS/JS portfolio website")
            self.console.print("    - Create a FastAPI app with authentication")
            self.console.print()
        
        # Chat Mode
        if mode == "chat" or mode is None:
            self.console.print("[bold cyan]CHAT MODE:[/bold cyan]")
            self.console.print("  Ask anything - coding help, explanations, debugging, etc.")
            self.console.print()
        
        # Analyze Mode
        if mode == "analyze" or mode is None:
            self.console.print("[bold yellow]ANALYZE MODE:[/bold yellow]")
            self.console.print("  analyze [path]     - Analyze directory/codebase")
            self.console.print("  search <pattern>   - Search for code patterns")
            self.console.print()
        
        # General
        self.console.print("[bold white]GENERAL:[/bold white]")
        self.console.print("  /help      - Show this help")
        self.console.print("  /exit      - Exit the agent")
        self.console.print()
        
        self.console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")
    
    def handle_model_change(self):
        """Handle model switching"""
        installed = self.get_installed_models()
        
        self.console.print("\n[bold cyan]AVAILABLE MODELS[/bold cyan]\n")
        
        # Show installed models
        if installed:
            self.console.print("[bold green]Installed:[/bold green]")
            for i, model in enumerate(installed, 1):
                current = " [yellow]<- current[/yellow]" if model == self.current_model else ""
                self.console.print(f"  {i}. {model}{current}")
            self.console.print()
        else:
            self.console.print("[yellow]No models installed yet[/yellow]\n")
        
        # Show available models
        self.console.print("[bold yellow]Available to install:[/bold yellow]")
        for i, model in enumerate(FREE_MODELS, 1):
            model_name = model.get("name", "unknown")
            model_desc = model.get("description", "")
            if model_name not in installed:
                self.console.print(f"  {i}. {model_name} - {model_desc}")
        
        self.console.print()
        
        choice = Prompt.ask("[cyan]Select model number or 'cancel'[/cyan]")
        
        if choice.lower() == 'cancel':
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(installed):
                self.current_model = installed[idx]
                self.console.print(f"\n[green]SUCCESS: Switched to {self.current_model}[/green]\n")
            elif 0 <= idx < len(FREE_MODELS):
                new_model = self.select_model()
                if new_model:
                    self.current_model = new_model
        except ValueError:
            self.console.print("[red]Invalid choice[/red]")
    
    def handle_analyze_mode(self):
        """Handle analyze mode interactions"""
        while self.mode == "analyze":
            try:
                user_input = Prompt.ask(f"[bold black on yellow] ANALYZE [/bold black on yellow] >>>")
                
                if user_input.lower() in ['exit', '/exit']:
                    return 'exit'
                
                if user_input.lower() in ['chat', '/chat']:
                    self.mode = "chat"
                    return None
                    
                if user_input.lower() in ['build', '/build']:
                    self.mode = "build"
                    return None
                
                if user_input.lower() in ['/help', 'help']:
                    self.show_help('analyze')
                    continue
                
                if user_input.lower().startswith('analyze'):
                    parts = user_input.split(maxsplit=1)
                    target = Path(parts[1]) if len(parts) > 1 else None
                    
                    self.console.print("\n[cyan]Analyzing codebase...[/cyan]\n")
                    analysis = self.analyzer.analyze_directory(target)
                    self.analyzer.display_analysis(analysis)
                
                elif user_input.lower().startswith('search '):
                    pattern = user_input[7:].strip()
                    if pattern:
                        self.console.print(f"\n[cyan]Searching for: '{pattern}'[/cyan]\n")
                        results = self.analyzer.search_code(pattern)
                        
                        if results:
                            table = Table(box=box.ROUNDED, border_style="yellow")
                            table.add_column("File", style="cyan")
                            table.add_column("Line", style="magenta", justify="right")
                            table.add_column("Content", style="white")
                            
                            for result in results[:50]:
                                table.add_row(
                                    result["file"],
                                    str(result["line"]),
                                    result["content"][:80]
                                )
                            
                            self.console.print(table)
                            self.console.print(f"\n[yellow]Found {len(results)} matches{' (showing first 50)' if len(results) > 50 else ''}[/yellow]\n")
                        else:
                            self.console.print("[yellow]No matches found[/yellow]\n")
                
            except KeyboardInterrupt:
                self.console.print()
                continue
    
    def start(self):
        """Enhanced main loop with mode selection"""
        self.display_animated_intro()
        
        if not self.check_prerequisites():
            return
        
        # Check for installed models first
        installed = self.get_installed_models()
        
        if installed:
            self.console.print("[bold cyan]>> Installed models:[/bold cyan]\n")
            
            table = Table(box=box.ROUNDED, border_style="cyan")
            table.add_column("#", style="cyan", width=5)
            table.add_column("Model", style="green")
            
            for idx, model in enumerate(installed, 1):
                table.add_row(str(idx), model)
            
            self.console.print(table)
            self.console.print()
            
            if Confirm.ask("[cyan]Use installed model?[/cyan]", default=True):
                if len(installed) == 1:
                    self.current_model = installed[0]
                else:
                    choice = Prompt.ask("[cyan]Select[/cyan]", choices=[str(i) for i in range(1, len(installed) + 1)])
                    self.current_model = installed[int(choice) - 1]
        
        # If no installed model selected, show available models
        if not self.current_model:
            self.console.print("\n[yellow]No models installed. Please select a model to install:[/yellow]\n")
            self.display_model_selection_menu()
            self.current_model = self.select_model()
        
        if not self.current_model:
            self.console.print("\n[red]No model selected. Exiting...[/red]\n")
            return
        
        # Main interaction loop
        self.console.print(f"\n[bold green]AGENT ACTIVE[/bold green]")
        self.console.print(f"[dim]Model: {self.current_model}[/dim]\n")
        
        self.show_mode_menu()
        
        while True:
            try:
                # Get mode-specific prompt
                if self.mode == "chat":
                    prompt_text = f"[bold black on cyan] CHAT [/bold black on cyan] >>>"
                elif self.mode == "build":
                    prompt_text = f"[bold black on magenta] BUILD [/bold black on magenta] >>>"
                else:  # analyze
                    if self.handle_analyze_mode() == 'exit':
                        break
                    self.show_mode_menu()
                    continue
                
                user_input = Prompt.ask(prompt_text)
                
                # Handle /help
                if user_input.lower() in ['/help', 'help']:
                    self.show_help(self.mode)
                    continue
                
                # Handle exit
                if user_input.lower() in ['exit', 'quit', '/exit', '/quit']:
                    self.console.print("\n[green]Goodbye![/green]\n")
                    break
                
                # Handle model switching
                if user_input.lower() in ['/model', '/models']:
                    self.handle_model_change()
                    continue
                
                # Handle mode switching with /commands
                if user_input.lower() in ['chat', '/chat']:
                    self.mode = "chat"
                    self.show_mode_menu()
                    continue
                elif user_input.lower() in ['build', '/build']:
                    self.mode = "build"
                    self.show_mode_menu()
                    continue
                elif user_input.lower() in ['analyze', '/analyze']:
                    self.mode = "analyze"
                    self.show_mode_menu()
                    continue
                
                if not user_input.strip():
                    continue
                
                # Process based on mode
                if self.mode == "build":
                    with self.show_thinking_animation() as progress:
                        task = progress.add_task("[cyan]Planning your build...", total=None)
                        response = self.chat_with_model_for_build(self.current_model, user_input)
                    
                    self.console.print()
                    
                    if response:
                        # Try to parse and create files
                        if self.parse_file_operations(response):
                            # Show what will be created with animation
                            self.console.print("[bold magenta]FILES TO CREATE:[/bold magenta]\n")
                            tree = self.filesystem.get_preview_tree()
                            self.console.print(tree)
                            self.console.print()
                            
                            if Confirm.ask("[magenta]Create these files?[/magenta]", default=True):
                                self.filesystem.execute_operations(self.console)
                            else:
                                self.console.print("[yellow]Cancelled file creation[/yellow]\n")
                                self.filesystem.clear_pending()
                        else:
                            # Just show response as conversation
                            self.console.print(Panel(
                                response,
                                title="[bold magenta]Response[/bold magenta]",
                                border_style="magenta"
                            ))
                            self.console.print()
                
                else:  # chat mode
                    with self.show_thinking_animation() as progress:
                        task = progress.add_task("[cyan]Thinking...", total=None)
                        response = self.chat_with_model(self.current_model, user_input)
                    
                    self.console.print()
                    
                    if response:
                        self.console.print(Panel(
                            response,
                            title="[bold cyan]Response[/bold cyan]",
                            border_style="cyan"
                        ))
                        self.console.print()
                        
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit[/yellow]\n")
                continue
            except Exception as e:
                self.console.print(f"\n[red]Error: {str(e)}[/red]\n")


if __name__ == "__main__":
    try:
        agent = LocalAgent()
        agent.start()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)