
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.align import Align
import time


LOGO_BLOCKS = r"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     █████╗ ███████╗ ██████╗ ██╗███████╗    ███████╗██╗     ██╗   ██╗██╗  ██╗   ║
    ║    ██╔══██╗██╔════╝██╔════╝ ██║██╔════╝    ██╔════╝██║     ██║   ██║╚██╗██╔╝   ║
    ║    ███████║█████╗  ██║  ███╗██║███████╗    █████╗  ██║     ██║   ██║ ╚███╔╝    ║
    ║    ██╔══██║██╔══╝  ██║   ██║██║╚════██║    ██╔══╝  ██║     ██║   ██║ ██╔██╗    ║
    ║    ██║  ██║███████╗╚██████╔╝██║███████║    ██║     ███████╗╚██████╔╝██╔╝ ██╗   ║
    ║    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝    ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
"""

LOGO_COMPACT = r"""
 ▄▀█ █▀▀ █▀▀ █ █▀   █▀▀ █   █ █ ▀▄▀
 █▀█ ██▄ █▄█ █ ▄█   █▀  █▄▄ █▄█ █ █
"""

LOGO_STYLED = [
    "    ░█████╗░███████╗░██████╗░██╗░██████╗  ███████╗██╗░░░░░██╗░░░██╗██╗░░██╗",
    "    ██╔══██╗██╔════╝██╔════╝░██║██╔════╝  ██╔════╝██║░░░░░██║░░░██║╚██╗██╔╝",
    "    ███████║█████╗░░██║░░██╗░██║╚█████╗░  █████╗░░██║░░░░░██║░░░██║░╚███╔╝░",
    "    ██╔══██║██╔══╝░░██║░░╚██╗██║░╚═══██╗  ██╔══╝░░██║░░░░░██║░░░██║░██╔██╗░",
    "    ██║░░██║███████╗╚██████╔╝██║██████╔╝  ██║░░░░░███████╗╚██████╔╝██╔╝╚██╗",
    "    ╚═╝░░╚═╝╚══════╝░╚═════╝░╚═╝╚═════╝░  ╚═╝░░░░░╚══════╝░╚═════╝░╚═╝░░╚═╝",
]

TAGLINE = "Multi-Agent AI Investment Analysis System"
VERSION = "v1.0.0"

COLORS = {
    "primary": "#00d4ff",
    "secondary": "#8b5cf6", 
    "accent": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "muted": "#6b7280",
}

AGENT_COLORS = {
    "data_scout": "#3b82f6",
    "technical_analyst": "#8b5cf6",
    "fundamental_analyst": "#10b981",
    "sentiment_analyst": "#f59e0b",
    "portfolio_manager": "#ef4444",
    "consensus_debate": "#ec4899",
    "guardrails": "#6366f1",
}

AGENT_ICONS = {
    "data_scout": ">>",
    "technical_analyst": "##",
    "fundamental_analyst": "$$",
    "sentiment_analyst": "@@",
    "portfolio_manager": "**",
    "consensus_debate": "<>",
    "guardrails": "[]",
}


def get_gradient_colors():
    """Get gradient colors for logo animation."""
    return [
        "#00d4ff", "#00c4ef", "#00b4df", "#00a4cf", 
        "#0094bf", "#0084af", "#00749f", "#00648f",
        "#00749f", "#0084af", "#0094bf", "#00a4cf",
        "#00b4df", "#00c4ef", "#00d4ff",
    ]


def render_logo_animated(console: Console, speed: float = 0.02):
    """Render the logo with animation effect."""
    console.clear()
    
    gradient = get_gradient_colors()
    
    for i, line in enumerate(LOGO_STYLED):
        color_idx = i % len(gradient)
        text = Text(line, style=f"bold {gradient[color_idx]}")
        console.print(text, justify="center")
        time.sleep(speed)
    
    console.print()
    
    tagline = Text()
    for i, char in enumerate(TAGLINE):
        color_idx = i % len(gradient)
        tagline.append(char, style=f"{gradient[color_idx]}")
    console.print(Align.center(tagline))
    
    console.print()
    version_text = Text(f"[ {VERSION} ]", style=f"dim {COLORS['muted']}")
    console.print(Align.center(version_text))
    console.print()


def render_logo_static(console: Console):
    """Render the logo without animation."""
    logo_text = Text()
    for line in LOGO_STYLED:
        logo_text.append(line + "\n", style=f"bold {COLORS['primary']}")
    
    console.print(Align.center(logo_text))
    console.print(Align.center(Text(TAGLINE, style=f"{COLORS['secondary']}")))
    console.print(Align.center(Text(f"[ {VERSION} ]", style=f"dim {COLORS['muted']}")))
    console.print()


def render_mini_logo(console: Console):
    """Render a compact logo for inline display."""
    text = Text()
    text.append("░▒▓", style=f"bold {COLORS['primary']}")
    text.append(" AEGIS FLUX ", style=f"bold white on {COLORS['primary']}")
    text.append("▓▒░", style=f"bold {COLORS['primary']}")
    console.print(text)


def get_agent_style(agent_name: str) -> tuple[str, str]:
    """Get color and icon for an agent."""
    agent_key = agent_name.lower().replace(" ", "_")
    color = AGENT_COLORS.get(agent_key, COLORS["muted"])
    icon = AGENT_ICONS.get(agent_key, "->")
    return color, icon


def create_header_panel(title: str, subtitle: str = "") -> Panel:
    """Create a styled header panel."""
    content = Text()
    content.append(title, style=f"bold {COLORS['primary']}")
    if subtitle:
        content.append(f"\n{subtitle}", style=f"dim {COLORS['muted']}")
    
    return Panel(
        Align.center(content),
        border_style=COLORS["primary"],
        padding=(1, 2),
    )


def create_section_header(title: str) -> Text:
    """Create a styled section header."""
    text = Text()
    text.append("━" * 3 + " ", style=COLORS["primary"])
    text.append(title.upper(), style=f"bold {COLORS['primary']}")
    text.append(" " + "━" * 50, style=COLORS["primary"])
    return text
