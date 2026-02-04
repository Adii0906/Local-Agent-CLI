#!/usr/bin/env python3
"""
LocalAgent SDK - Open Source Terminal AI Agent
Enhanced with Model Management & Code Analysis
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Callable
import subprocess
import re
import os

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
    from rich.markdown import Markdown
    from rich.layout import Layout
    from rich.live import Live
except ImportError:
    print("Error: 'rich' library not found!")
    print("\nInstall: pip install rich")
    sys.exit(1)

from ascii_art import BANNER, SUBTITLE
from models import FREE_MODELS

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
        """Generate preview tree"""
        tree = Tree("[bold cyan]📁 Project Structure[/bold cyan]", guide_style="cyan")
        
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
                    node = current.add(f"📁 [cyan]{part}[/cyan] {status}")
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
            current.add(f"📄 [white]{parts[-1]}[/white]{size} {status}")
        
        return tree
    
    def execute_operations(self, console: Console) -> bool:
        """Execute operations with progress"""
        if not self.pending_operations:
            return True
        
        overwrites = [op for op in self.pending_operations if op.exists and not op.is_dir]
        
        if overwrites:
            console.print("\n[bold yellow]⚠️  Files exist:[/bold yellow]")
            for op in overwrites:
                rel = op.path.relative_to(self.workspace)
                console.print(f"  • {rel}")
            console.print()
            if not Confirm.ask("[yellow]Overwrite?[/yellow]", default=False):
                console.print("[red]✗ Cancelled[/red]\n")
                return False
        
        try:
            # Show what we're creating
            console.print("\n[cyan]Creating files...[/cyan]\n")
            
            for op in self.pending_operations:
                if op.is_dir:
                    op.path.mkdir(parents=True, exist_ok=True)
                    console.print(f"[green]✓[/green] Created folder: [cyan]{op.path.name}[/cyan]")
                else:
                    op.path.parent.mkdir(parents=True, exist_ok=True)
                    op.path.write_text(op.content, encoding='utf-8')
                    console.print(f"[green]✓[/green] Wrote file: [cyan]{op.path.name}[/cyan] ({len(op.content)} bytes)")
                time.sleep(0.1)
            
            console.print(f"\n[bold green]✓ Success! Created {len(self.pending_operations)} items[/bold green]\n")
            return True
            
        except Exception as e:
            console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
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
            "total_lines": 0,
            "total_size": 0,
            "file_types": {}
        }
        
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build', '.next'}
        
        for item in target_path.rglob('*'):
            # Skip excluded directories
            if any(excluded in item.parts for excluded in exclude_dirs):
                continue
            
            if item.is_file():
                try:
                    size = item.stat().st_size
                    analysis["total_size"] += size
                    
                    ext = item.suffix.lower()
                    analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1
                    
                    # Count lines for text files
                    if ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.html', '.css', '.md', '.txt']:
                        try:
                            with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = sum(1 for _ in f)
                                analysis["total_lines"] += lines
                                
                            # Language detection
                            lang = self._detect_language(ext)
                            if lang:
                                if lang not in analysis["languages"]:
                                    analysis["languages"][lang] = {"files": 0, "lines": 0}
                                analysis["languages"][lang]["files"] += 1
                                analysis["languages"][lang]["lines"] += lines
                        except:
                            pass
                    
                    analysis["files"].append({
                        "path": str(item.relative_to(self.workspace)),
                        "size": size,
                        "ext": ext
                    })
                except:
                    continue
            elif item.is_dir():
                analysis["directories"].append(str(item.relative_to(self.workspace)))
        
        return analysis
    
    def _detect_language(self, ext: str) -> Optional[str]:
        """Detect programming language from extension"""
        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React/JSX',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript/React',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.html': 'HTML',
            '.css': 'CSS',
            '.md': 'Markdown'
        }
        return lang_map.get(ext)
    
    def display_analysis(self, analysis: Dict):
        """Display analysis with rich formatting"""
        # Summary Panel
        total_files = len(analysis["files"])
        total_dirs = len(analysis["directories"])
        total_size_mb = analysis["total_size"] / (1024 * 1024)
        
        summary = Panel(
            f"[cyan]Files:[/cyan] {total_files}\n"
            f"[cyan]Directories:[/cyan] {total_dirs}\n"
            f"[cyan]Total Lines:[/cyan] {analysis['total_lines']:,}\n"
            f"[cyan]Total Size:[/cyan] {total_size_mb:.2f} MB",
            title="[bold cyan]📊 Codebase Summary[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )
        self.console.print(summary)
        self.console.print()
        
        # Languages Table
        if analysis["languages"]:
            self.console.print("[bold cyan]📝 Languages:[/bold cyan]\n")
            table = Table(box=box.ROUNDED, border_style="cyan")
            table.add_column("Language", style="green")
            table.add_column("Files", style="yellow", justify="right")
            table.add_column("Lines", style="magenta", justify="right")
            
            sorted_langs = sorted(analysis["languages"].items(), key=lambda x: x[1]["lines"], reverse=True)
            for lang, stats in sorted_langs:
                table.add_row(lang, str(stats["files"]), f"{stats['lines']:,}")
            
            self.console.print(table)
            self.console.print()
        
        # File Types
        if analysis["file_types"]:
            self.console.print("[bold cyan]📁 File Types:[/bold cyan]\n")
            table = Table(box=box.ROUNDED, border_style="cyan")
            table.add_column("Extension", style="green")
            table.add_column("Count", style="yellow", justify="right")
            
            sorted_types = sorted(analysis["file_types"].items(), key=lambda x: x[1], reverse=True)[:10]
            for ext, count in sorted_types:
                table.add_row(ext if ext else "(no ext)", str(count))
            
            self.console.print(table)
            self.console.print()
    
    def search_code(self, pattern: str, file_pattern: str = "*") -> List[Dict]:
        """Search for pattern in codebase"""
        results = []
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
        
        for file_path in self.workspace.rglob(file_pattern):
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                results.append({
                                    "file": str(file_path.relative_to(self.workspace)),
                                    "line": line_num,
                                    "content": line.strip()
                                })
                except:
                    continue
        
        return results


class LocalAgent:
    def __init__(self):
        self.console = console
        self.current_model = None
        self.conversation_history = []
        self.filesystem = SafeFileSystem(WORKSPACE_DIR)
        self.analyzer = CodeAnalyzer(WORKSPACE_DIR)
        self.mode = "chat"  # chat, build, or analyze
    
    def show_thinking_animation(self, message: str = "Thinking"):
        """Show animated thinking spinner"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}[/cyan]"),
            console=self.console,
            transient=True
        )
    
    def show_help(self, mode: str):
        """Show help for current mode"""
        if mode == "chat":
            help_text = """[bold cyan]CHAT MODE - Help[/bold cyan]

[yellow]What you can do:[/yellow]
• Ask questions and get AI responses
• Get code explanations
• Debug help
• General conversation

[yellow]Commands:[/yellow]
• [cyan]/build[/cyan] - Switch to build mode
• [cyan]/analyze[/cyan] - Switch to analyze mode
• [cyan]/help[/cyan] - Show this help
• [cyan]/exit[/cyan] - Quit application

[yellow]Examples:[/yellow]
• Explain how decorators work in Python
• What's the difference between let and const?
• Help me debug this error: [paste error]
"""
        elif mode == "build":
            help_text = """[bold magenta]BUILD MODE - Help[/bold magenta]

[yellow]What you can do:[/yellow]
• Create files and folders
• Generate complete projects
• Build components and scripts
• Setup project structures

[yellow]Commands:[/yellow]
• [cyan]/chat[/cyan] - Switch to chat mode
• [cyan]/analyze[/cyan] - Switch to analyze mode
• [cyan]/help[/cyan] - Show this help
• [cyan]/exit[/cyan] - Quit application

[yellow]Examples:[/yellow]
• Create a Flask REST API with authentication
• Build a React todo component
• Make a Python script to scrape news
• Setup a Node.js Express server

[yellow]Tips:[/yellow]
• Be specific about what files you want
• Use words like 'create', 'build', 'make'
• If just chatting, it won't create files
"""
        else:  # analyze
            help_text = """[bold yellow]ANALYZE MODE - Help[/bold yellow]

[yellow]What you can do:[/yellow]
• Analyze your entire codebase
• Get statistics and metrics
• Search for code patterns
• Find specific functions or classes

[yellow]Commands:[/yellow]
• [cyan]analyze[/cyan] - Analyze entire codebase
• [cyan]analyze <path>[/cyan] - Analyze specific folder
• [cyan]search <pattern>[/cyan] - Search in code
• [cyan]/chat[/cyan] - Switch to chat mode
• [cyan]/build[/cyan] - Switch to build mode
• [cyan]/help[/cyan] - Show this help
• [cyan]/exit[/cyan] - Quit application

[yellow]Examples:[/yellow]
• analyze
• analyze src
• search TODO
• search function login
"""
        
        self.console.print(Panel(help_text, border_style="cyan", padding=(1, 2)))
        self.console.print()
    
    def check_ollama_installed(self) -> bool:
        """Check Ollama"""
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def check_model_installed(self, model_name: str) -> bool:
        """Check if model installed"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False, timeout=10, encoding='utf-8', errors='replace')
            return model_name in result.stdout
        except Exception:
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get installed models"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False, timeout=10, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
        except Exception:
            pass
        return []
    
    def download_model(self, model_name: str) -> bool:
        """Download model with progress"""
        self.console.print(f"\n[bold cyan]📥 Downloading {model_name}[/bold cyan]\n")
        
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Downloading {model_name}", total=100)
                
                for line in process.stdout:
                    if "%" in line:
                        try:
                            percent = re.search(r'(\d+)%', line)
                            if percent:
                                progress.update(task, completed=int(percent.group(1)))
                        except:
                            pass
                    progress.update(task, description=line.strip()[:50])
            
            process.wait()
            
            if process.returncode == 0:
                self.console.print(f"\n[bold green]✓ Successfully downloaded {model_name}[/bold green]\n")
                return True
            else:
                self.console.print(f"\n[bold red]✗ Download failed[/bold red]\n")
                return False
                
        except Exception as e:
            self.console.print(f"\n[bold red]✗ Error: {str(e)}[/bold red]\n")
            return False
    
    def chat_with_model(self, model_name: str, user_message: str) -> Optional[str]:
        """Chat with model"""
        try:
            process = subprocess.Popen(
                ["ollama", "run", model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout, stderr = process.communicate(input=user_message, timeout=120)
            
            if process.returncode == 0 and stdout.strip():
                response = stdout.strip()
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
            else:
                return None
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]Request timed out[/red]")
            process.kill()
            return None
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            return None
    
    def chat_with_model_for_build(self, model_name: str, user_message: str) -> Optional[str]:
        """Chat for build mode with file operation instructions"""
        
        # Check if user is actually asking to create files
        create_keywords = ['create', 'make', 'build', 'generate', 'write', 'add', 'new file', 'setup']
        should_create_files = any(keyword in user_message.lower() for keyword in create_keywords)
        
        if not should_create_files:
            # Just chat normally without trying to create files
            return self.chat_with_model(model_name, user_message)
        
        enhanced_prompt = f"""You are a code generation assistant. Create files based on the user's request.

User request: {user_message}

IMPORTANT: You MUST respond in this exact format:

FOLDER: folder_name
FILE: filename.ext
```
[complete file content here]
```

Example response format:
FOLDER: src
FILE: src/app.py
```python
print("Hello World")
```

FILE: README.md
```markdown
# My Project
```

Now create the files for the user's request."""
        
        try:
            process = subprocess.Popen(
                ["ollama", "run", model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout, stderr = process.communicate(input=enhanced_prompt, timeout=120)
            
            if process.returncode == 0 and stdout.strip():
                response = stdout.strip()
                return response
            else:
                return None
                
        except subprocess.TimeoutExpired:
            self.console.print("[red]Request timed out[/red]")
            process.kill()
            return None
        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
            return None
    
    def parse_file_operations(self, response: str) -> bool:
        """Parse response for file operations"""
        self.filesystem.clear_pending()
        found_operations = False
        
        # Find folders - more flexible pattern
        folder_patterns = [
            r'FOLDER:\s*(.+)',
            r'Folder:\s*(.+)',
            r'folder:\s*(.+)',
            r'Directory:\s*(.+)',
        ]
        
        for pattern in folder_patterns:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                folder_path = match.group(1).strip()
                if self.filesystem.plan_folder(folder_path):
                    found_operations = True
        
        # Find files with content - multiple patterns
        file_patterns = [
            r'FILE:\s*([^\n]+)\n```(?:\w+)?\n(.*?)```',
            r'File:\s*([^\n]+)\n```(?:\w+)?\n(.*?)```',
            r'file:\s*([^\n]+)\n```(?:\w+)?\n(.*?)```',
            # Also try without FILE: prefix if we see code blocks with filenames
            r'`([^\s`]+\.\w+)`\n```(?:\w+)?\n(.*?)```',
        ]
        
        for pattern in file_patterns:
            for match in re.finditer(pattern, response, re.DOTALL | re.IGNORECASE):
                file_path = match.group(1).strip()
                content = match.group(2).strip()
                if self.filesystem.plan_file(file_path, content):
                    found_operations = True
        
        return found_operations
    
    def display_animated_intro(self):
        """Display animated intro"""
        self.console.clear()
        self.console.print(f"{BANNER}", justify="left")
        self.console.print(f"{SUBTITLE}", justify="left")
        self.console.print()
        
        for countdown in range(3, 0, -1):
            self.console.print(f"[cyan]Initializing... {countdown}s[/cyan]", end="\r")
            time.sleep(1)
        
        self.console.print(" " * 50, end="\r")
        self.console.print("[green]✓ Ready![/green]")
        time.sleep(0.5)
        self.console.print()
    
    def check_prerequisites(self) -> bool:
        """Check prerequisites"""
        self.console.print("[bold cyan]>> Checking system[/bold cyan]\n")
        
        if not self.check_ollama_installed():
            self.console.print("[bold red]✗ Ollama not found[/bold red]\n")
            
            error = Panel(
                "[yellow]Ollama is required[/yellow]\n\n"
                "[white]macOS:[/white]  brew install ollama\n"
                "[white]Linux:[/white]  curl -fsSL https://ollama.ai/install.sh | sh\n"
                "[white]Windows:[/white]  Visit https://ollama.ai",
                title="Missing: Ollama",
                border_style="yellow",
                padding=(1, 2)
            )
            self.console.print(error)
            return False
        
        self.console.print("[bold green]✓ Ollama ready[/bold green]\n")
        time.sleep(0.5)
        return True
    
    def display_model_selection_menu(self):
        """Display enhanced model selection with color-coded status"""
        self.console.print("[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/bold cyan]")
        self.console.print("[bold cyan]║              🤖  MODEL SELECTION & MANAGEMENT  🤖              ║[/bold cyan]")
        self.console.print("[bold cyan]╚═══════════════════════════════════════════════════════════════╝[/bold cyan]\n")
        
        installed = self.get_installed_models()
        
        table = Table(box=box.DOUBLE_EDGE, border_style="cyan", header_style="bold magenta")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Model", style="white", width=22)
        table.add_column("Description", style="dim white", width=32)
        table.add_column("Size", style="yellow", width=10)
        table.add_column("Status", style="white", width=15)
        
        for key, model in FREE_MODELS.items():
            name = model['name']
            is_installed = name in installed
            
            # Add star for recommended
            display_name = f"⭐ {name}" if model['recommended'] else name
            
            # Color-coded status
            if is_installed:
                status = "[green]● READY[/green]"
            else:
                status = "[red]○ NOT INSTALLED[/red]"
            
            table.add_row(
                key,
                display_name,
                model['description'],
                model['size'],
                status
            )
        
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Legend: ⭐ Recommended | [green]● Ready to use[/green] | [red]○ Needs download[/red][/dim]\n")
    
    def select_model(self) -> Optional[str]:
        """Enhanced model selection"""
        while True:
            choice = Prompt.ask(
                "[cyan]Select model number[/cyan]",
                choices=list(FREE_MODELS.keys()),
                default="1"
            )
            
            model = FREE_MODELS[choice]
            model_name = model["name"]
            
            self.console.print()
            
            if not self.check_model_installed(model_name):
                self.console.print(Panel(
                    f"[yellow]Model:[/yellow] {model_name}\n"
                    f"[yellow]Size:[/yellow] {model['size']}\n"
                    f"[yellow]Status:[/yellow] Not installed",
                    title="⚠️  Download Required",
                    border_style="yellow"
                ))
                self.console.print()
                
                if Confirm.ask(f"[cyan]Download {model_name}?[/cyan]", default=True):
                    if self.download_model(model_name):
                        return model_name
                    continue
                else:
                    if not Confirm.ask("[yellow]Try another model?[/yellow]", default=True):
                        return None
                    self.console.print()
                    self.display_model_selection_menu()
                    continue
            else:
                self.console.print(f"[green]✓ {model_name} is ready![/green]\n")
                return model_name
    
    def show_mode_menu(self):
        """Display mode selection menu"""
        mode_colors = {
            "chat": "cyan",
            "build": "magenta",
            "analyze": "yellow"
        }
        
        mode_icons = {
            "chat": "🗨",
            "build": "🔨",
            "analyze": "🔍"
        }
        
        color = mode_colors[self.mode]
        icon = mode_icons[self.mode]
        
        self.console.print(f"\n[bold {color}]{icon} Current Mode: {self.mode.upper()}[/bold {color}]")
        self.console.print("[yellow]Commands: /chat | /build | /analyze | /help | /exit[/yellow]\n")
    
    def handle_analyze_mode(self):
        """Handle code analysis mode"""
        self.console.print(Panel(
            "[bold]Code Analysis Commands:[/bold]\n\n"
            "[cyan]analyze[/cyan] - Analyze entire codebase\n"
            "[cyan]analyze <path>[/cyan] - Analyze specific directory\n"
            "[cyan]search <pattern>[/cyan] - Search for pattern in code\n"
            "[yellow]/chat[/yellow] - Switch to chat mode\n"
            "[yellow]/build[/yellow] - Switch to build mode\n"
            "[yellow]/exit[/yellow] - Quit application",
            title="[yellow]🔍 Analysis Mode[/yellow]",
            border_style="yellow",
            box=box.DOUBLE_EDGE
        ))
        self.console.print()
        
        while self.mode == "analyze":
            try:
                user_input = Prompt.ask(f"[bold yellow]analyze[/bold yellow]")
                
                if user_input.lower() in ['exit', '/exit']:
                    return 'exit'
                
                if user_input.lower() in ['chat', '/chat']:
                    self.mode = "chat"
                    return None
                    
                if user_input.lower() in ['build', '/build']:
                    self.mode = "build"
                    return None
                
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
                            
                            for result in results[:50]:  # Limit to 50 results
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
            
            if Confirm.ask("[cyan]Use installed?[/cyan]", default=True):
                if len(installed) == 1:
                    self.current_model = installed[0]
                else:
                    choice = Prompt.ask("[cyan]Select[/cyan]", choices=[str(i) for i in range(1, len(installed) + 1)])
                    self.current_model = installed[int(choice) - 1]
        
        # If no installed model selected, show available models
        if not self.current_model:
            self.display_model_selection_menu()
            self.current_model = self.select_model()
        
        if not self.current_model:
            return
        
        # Main interaction loop
        self.console.print(f"\n[bold green]✓ Agent Active[/bold green]")
        self.console.print(f"[dim]Model: {self.current_model}[/dim]\n")
        
        self.show_mode_menu()
        
        while True:
            try:
                # Get mode-specific prompt
                if self.mode == "chat":
                    prompt_text = f"[bold cyan]chat({self.current_model})[/bold cyan]"
                elif self.mode == "build":
                    prompt_text = f"[bold magenta]🔨 build[/bold magenta]"
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
                    self.console.print("\n[green]👋 Goodbye![/green]\n")
                    break
                
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
                        task = progress.add_task("Planning your build...", total=None)
                        response = self.chat_with_model_for_build(self.current_model, user_input)
                    
                    self.console.print()
                    
                    if response:
                        # Try to parse and create files
                        if self.parse_file_operations(response):
                            # Show what will be created
                            self.console.print("[bold magenta]📋 Files to create:[/bold magenta]\n")
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
                        task = progress.add_task("Thinking...", total=None)
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
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)