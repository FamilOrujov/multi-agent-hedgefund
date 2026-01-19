
import asyncio
import sys
import time
from contextlib import contextmanager
from typing import Optional, Generator

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table
from rich.align import Align

from src.cli.branding import COLORS, get_agent_style, AGENT_COLORS


SPINNER_FRAMES = {
    "blocks": ["▖", "▘", "▝", "▗"],
    "mining": ["⛏ ", " ⛏", "  ⛏", " ⛏"],
    "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "arrows": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    "pulse": ["░", "▒", "▓", "█", "▓", "▒"],
    "wave": ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"],
    "scan": ["[    ]", "[=   ]", "[==  ]", "[=== ]", "[====]", "[ ===]", "[  ==]", "[   =]"],
    "data": ["[>>  ]", "[ >> ]", "[  >>]", "[  <<]", "[ << ]", "[<<  ]"],
    "think": ["[.   ]", "[..  ]", "[... ]", "[....]", "[ ...]", "[  ..]", "[   .]"],
}


class AnimatedSpinner:
    """Custom animated spinner with multiple frame styles."""
    
    def __init__(
        self,
        console: Console,
        message: str = "Loading",
        style: str = "dots",
        color: str = COLORS["primary"],
    ):
        self.console = console
        self.message = message
        self.frames = SPINNER_FRAMES.get(style, SPINNER_FRAMES["dots"])
        self.color = color
        self.running = False
        self.frame_idx = 0
    
    def get_frame(self) -> Text:
        """Get current frame as Text."""
        frame = self.frames[self.frame_idx % len(self.frames)]
        self.frame_idx += 1
        
        text = Text()
        text.append(frame, style=f"bold {self.color}")
        text.append(f" {self.message}", style=self.color)
        return text


class AgentProgressDisplay:
    """Display agent progress with animations."""
    
    def __init__(self, console: Console):
        self.console = console
        self.agents_status: dict[str, dict] = {}
        self.current_thinking: dict[str, str] = {}
        self.live: Optional[Live] = None
    
    def create_display(self) -> Table:
        """Create the agent progress display table."""
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"],
            expand=True,
            padding=(0, 1),
        )
        
        table.add_column("Agent", style="bold", width=22)
        table.add_column("Status", width=12)
        table.add_column("Activity", ratio=1)
        
        agent_order = [
            "data_scout",
            "technical_analyst", 
            "fundamental_analyst",
            "sentiment_analyst",
            "consensus_debate",
            "portfolio_manager",
            "guardrails",
        ]
        
        for agent_key in agent_order:
            if agent_key in self.agents_status:
                status = self.agents_status[agent_key]
                color, icon = get_agent_style(agent_key)
                
                agent_name = agent_key.replace("_", " ").title()
                name_text = Text()
                name_text.append(f"{icon} ", style=f"bold {color}")
                name_text.append(agent_name, style=f"{color}")
                
                status_text = self._get_status_text(status["status"], color)
                
                activity = status.get("activity", "")
                thinking = self.current_thinking.get(agent_key, "")
                
                if thinking:
                    activity_text = Text(thinking[:50] + "..." if len(thinking) > 50 else thinking)
                    activity_text.stylize(f"italic {COLORS['muted']}")
                elif activity:
                    activity_text = Text(activity[:50] + "..." if len(activity) > 50 else activity)
                else:
                    activity_text = Text("-", style="dim")
                
                table.add_row(name_text, status_text, activity_text)
        
        return table
    
    def _get_status_text(self, status: str, color: str) -> Text:
        """Get styled status text."""
        text = Text()
        
        if status == "running":
            frames = SPINNER_FRAMES["pulse"]
            frame = frames[int(time.time() * 8) % len(frames)]
            text.append(f"{frame} ", style=f"bold {color}")
            text.append("Running", style=color)
        elif status == "done":
            text.append(">> ", style=f"bold {COLORS['accent']}")
            text.append("Done", style=COLORS["accent"])
        elif status == "error":
            text.append("!! ", style=f"bold {COLORS['danger']}")
            text.append("Error", style=COLORS["danger"])
        elif status == "waiting":
            text.append(".. ", style=f"dim {COLORS['muted']}")
            text.append("Waiting", style=f"dim {COLORS['muted']}")
        else:
            text.append("-- ", style="dim")
            text.append(status.title(), style="dim")
        
        return text
    
    def set_agent_status(self, agent: str, status: str, activity: str = ""):
        """Update an agent's status."""
        self.agents_status[agent] = {
            "status": status,
            "activity": activity,
        }
    
    def set_thinking(self, agent: str, content: str):
        """Set current thinking content for an agent."""
        self.current_thinking[agent] = content
    
    def clear_thinking(self, agent: str):
        """Clear thinking content for an agent."""
        if agent in self.current_thinking:
            del self.current_thinking[agent]


class StreamingText:
    """Animated streaming text display."""
    
    def __init__(
        self,
        console: Console,
        prefix: str = "",
        color: str = COLORS["primary"],
        typing_speed: float = 0.02,
    ):
        self.console = console
        self.prefix = prefix
        self.color = color
        self.typing_speed = typing_speed
        self.buffer = ""
    
    def stream_char(self, char: str):
        """Stream a single character with typing effect."""
        self.buffer += char
        self.console.print(char, end="", style=self.color)
        time.sleep(self.typing_speed)
    
    def stream_text(self, text: str, animate: bool = True):
        """Stream text with optional animation."""
        if animate:
            for char in text:
                self.stream_char(char)
        else:
            self.console.print(text, end="", style=self.color)
            self.buffer += text
    
    def newline(self):
        """Print newline."""
        self.console.print()
        self.buffer += "\n"
    
    def finish(self):
        """Finish streaming and return buffer."""
        self.console.print()
        return self.buffer


class ThinkingAnimation:
    """Animated thinking display for agents."""
    
    def __init__(self, console: Console, agent_name: str):
        self.console = console
        self.agent_name = agent_name
        self.color, self.icon = get_agent_style(agent_name)
        self.thoughts: list[str] = []
        self.frame_idx = 0
    
    def create_display(self) -> Panel:
        """Create thinking display panel."""
        content = Text()
        
        header = Text()
        frames = SPINNER_FRAMES["think"]
        frame = frames[self.frame_idx % len(frames)]
        self.frame_idx += 1
        
        header.append(f"{self.icon} ", style=f"bold {self.color}")
        header.append(self.agent_name.replace("_", " ").title(), style=f"bold {self.color}")
        header.append(f" {frame}", style=f"bold {self.color}")
        
        content.append(header)
        content.append("\n")
        
        for i, thought in enumerate(self.thoughts[-5:]):
            if i == len(self.thoughts[-5:]) - 1:
                content.append(f"  >> {thought}", style=f"{self.color}")
            else:
                content.append(f"     {thought}\n", style=f"dim {COLORS['muted']}")
        
        return Panel(
            content,
            border_style=self.color,
            padding=(0, 1),
        )
    
    def add_thought(self, thought: str):
        """Add a thought to the display."""
        self.thoughts.append(thought)


class ProgressAnimation:
    """Multi-step progress animation."""
    
    def __init__(self, console: Console, total_steps: int):
        self.console = console
        self.total_steps = total_steps
        self.current_step = 0
        self.step_names: list[str] = []
    
    def create_progress_bar(self) -> Text:
        """Create an animated progress bar."""
        width = 40
        filled = int((self.current_step / self.total_steps) * width)
        
        text = Text()
        text.append("[", style=COLORS["muted"])
        
        for i in range(width):
            if i < filled:
                text.append("█", style=f"bold {COLORS['primary']}")
            elif i == filled:
                frames = ["░", "▒", "▓"]
                frame = frames[int(time.time() * 4) % len(frames)]
                text.append(frame, style=f"bold {COLORS['primary']}")
            else:
                text.append("░", style=f"dim {COLORS['muted']}")
        
        text.append("]", style=COLORS["muted"])
        text.append(f" {self.current_step}/{self.total_steps}", style=COLORS["primary"])
        
        return text
    
    def advance(self, step_name: str = ""):
        """Advance to next step."""
        self.current_step = min(self.current_step + 1, self.total_steps)
        if step_name:
            self.step_names.append(step_name)


@contextmanager
def loading_animation(
    console: Console,
    message: str = "Processing",
    style: str = "dots",
) -> Generator[None, None, None]:
    """Context manager for loading animation."""
    spinner = AnimatedSpinner(console, message, style)
    
    with Live(spinner.get_frame(), console=console, refresh_per_second=10) as live:
        try:
            yield
            while True:
                live.update(spinner.get_frame())
                time.sleep(0.1)
        except GeneratorExit:
            pass


def print_success(console: Console, message: str):
    """Print a success message with animation."""
    text = Text()
    text.append(">> ", style=f"bold {COLORS['accent']}")
    text.append(message, style=COLORS["accent"])
    console.print(text)


def print_error(console: Console, message: str):
    """Print an error message."""
    text = Text()
    text.append("!! ", style=f"bold {COLORS['danger']}")
    text.append(message, style=COLORS["danger"])
    console.print(text)


def print_warning(console: Console, message: str):
    """Print a warning message."""
    text = Text()
    text.append("** ", style=f"bold {COLORS['warning']}")
    text.append(message, style=COLORS["warning"])
    console.print(text)


def print_info(console: Console, message: str):
    """Print an info message."""
    text = Text()
    text.append(":: ", style=f"bold {COLORS['primary']}")
    text.append(message, style=COLORS["primary"])
    console.print(text)


async def animate_text_stream(console: Console, text: str, speed: float = 0.01):
    """Async animated text streaming."""
    for char in text:
        console.print(char, end="")
        await asyncio.sleep(speed)
    console.print()
