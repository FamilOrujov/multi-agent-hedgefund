
import asyncio
import time
from typing import Optional, Any
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.box import ROUNDED, HEAVY, DOUBLE

from src.cli.branding import COLORS, get_agent_style, create_section_header
from src.cli.animations import SPINNER_FRAMES


@dataclass
class AgentState:
    """State for a single agent."""
    name: str
    status: str = "pending"
    thinking: str = ""
    signal: Optional[str] = None
    confidence: Optional[int] = None
    summary: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class DebateState:
    """State for consensus debate."""
    active: bool = False
    round_num: int = 0
    positions: dict = field(default_factory=dict)
    arguments: list = field(default_factory=list)
    result: Optional[str] = None


class StreamingAgentDisplay:
    """Real-time streaming display for multi-agent workflow."""
    
    def __init__(self, console: Console, ticker: str):
        self.console = console
        self.ticker = ticker
        self.agents: dict[str, AgentState] = {}
        self.debate = DebateState()
        self.final_decision: Optional[dict] = None
        self.requires_review = False
        self.review_triggers: list[str] = []
        self.frame_idx = 0
        self.messages: list[tuple[str, str]] = []
        
        self._init_agents()
    
    def _init_agents(self):
        """Initialize agent states."""
        agent_names = [
            "data_scout",
            "technical_analyst",
            "fundamental_analyst", 
            "sentiment_analyst",
            "consensus_debate",
            "portfolio_manager",
            "guardrails",
        ]
        for name in agent_names:
            self.agents[name] = AgentState(name=name)
    
    def create_header(self) -> Panel:
        """Create the analysis header."""
        content = Text()
        content.append("ANALYZING: ", style=f"bold {COLORS['muted']}")
        content.append(self.ticker, style=f"bold {COLORS['primary']}")
        
        return Panel(
            Align.center(content),
            border_style=COLORS["primary"],
            padding=(0, 2),
        )
    
    def create_agents_panel(self) -> Panel:
        """Create the agents status panel."""
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"],
            expand=True,
            box=ROUNDED,
            padding=(0, 1),
        )
        
        table.add_column("Agent", width=24)
        table.add_column("Status", width=14)
        table.add_column("Signal", width=12)
        table.add_column("Activity", ratio=1)
        
        display_order = [
            "data_scout",
            "technical_analyst",
            "fundamental_analyst",
            "sentiment_analyst",
            "consensus_debate",
            "portfolio_manager",
            "guardrails",
        ]
        
        for agent_key in display_order:
            if agent_key not in self.agents:
                continue
                
            agent = self.agents[agent_key]
            color, icon = get_agent_style(agent_key)
            
            name_text = Text()
            name_text.append(f"{icon} ", style=f"bold {color}")
            name_text.append(agent.name.replace("_", " ").title(), style=color)
            
            status_text = self._create_status_text(agent.status, color)
            
            signal_text = Text("-", style="dim")
            if agent.signal:
                sig_color = self._get_signal_color(agent.signal)
                signal_text = Text(agent.signal, style=f"bold {sig_color}")
                if agent.confidence:
                    signal_text.append(f" {agent.confidence}%", style=f"dim {sig_color}")
            
            activity = agent.thinking or agent.summary or "-"
            if len(activity) > 45:
                activity = activity[:42] + "..."
            activity_text = Text(activity, style=f"italic {COLORS['muted']}" if agent.thinking else "dim")
            
            table.add_row(name_text, status_text, signal_text, activity_text)
        
        return Panel(
            table,
            title="[bold]Agent Pipeline[/bold]",
            title_align="left",
            border_style=COLORS["primary"],
            padding=(0, 0),
        )
    
    def _create_status_text(self, status: str, color: str) -> Text:
        """Create animated status text."""
        text = Text()
        
        if status == "running":
            frames = SPINNER_FRAMES["pulse"]
            frame = frames[self.frame_idx % len(frames)]
            text.append(f"{frame} ", style=f"bold {color}")
            text.append("Running", style=color)
        elif status == "done":
            text.append(">> ", style=f"bold {COLORS['accent']}")
            text.append("Complete", style=COLORS["accent"])
        elif status == "error":
            text.append("!! ", style=f"bold {COLORS['danger']}")
            text.append("Error", style=COLORS["danger"])
        elif status == "pending":
            text.append(".. ", style=f"dim {COLORS['muted']}")
            text.append("Waiting", style=f"dim {COLORS['muted']}")
        else:
            text.append("-- ", style="dim")
            text.append(status.title(), style="dim")
        
        return text
    
    def _get_signal_color(self, signal: str) -> str:
        """Get color for a signal."""
        signal_lower = signal.lower()
        if "bull" in signal_lower or "buy" in signal_lower:
            return COLORS["accent"]
        elif "bear" in signal_lower or "sell" in signal_lower:
            return COLORS["danger"]
        return COLORS["warning"]
    
    def create_debate_panel(self) -> Optional[Panel]:
        """Create debate panel if debate is active."""
        if not self.debate.active:
            return None
        
        content = Text()
        
        header = Text()
        frames = SPINNER_FRAMES["data"]
        frame = frames[self.frame_idx % len(frames)]
        header.append(f"{frame} ", style=f"bold {COLORS['secondary']}")
        header.append("CONSENSUS DEBATE", style=f"bold {COLORS['secondary']}")
        header.append(f" - Round {self.debate.round_num}", style=COLORS["muted"])
        content.append(header)
        content.append("\n\n")
        
        if self.debate.positions:
            content.append("Positions:\n", style=f"bold {COLORS['muted']}")
            for agent, pos in self.debate.positions.items():
                pos_color = self._get_signal_color(pos)
                content.append(f"  {agent}: ", style=COLORS["muted"])
                content.append(f"{pos}\n", style=f"bold {pos_color}")
        
        if self.debate.arguments:
            content.append("\nLatest Arguments:\n", style=f"bold {COLORS['muted']}")
            for arg in self.debate.arguments[-3:]:
                agent = arg.get("agent", "")
                msg = arg.get("message", "")[:60]
                content.append(f"  {agent}: ", style=COLORS["secondary"])
                content.append(f"{msg}...\n", style=f"italic {COLORS['muted']}")
        
        if self.debate.result:
            content.append("\n")
            content.append(">> Result: ", style=f"bold {COLORS['accent']}")
            content.append(self.debate.result, style=f"bold {COLORS['accent']}")
        
        return Panel(
            content,
            title="[bold]Team Discussion[/bold]",
            title_align="left",
            border_style=COLORS["secondary"],
            padding=(0, 1),
        )
    
    def create_decision_panel(self) -> Optional[Panel]:
        """Create final decision panel."""
        if not self.final_decision:
            return None
        
        decision = self.final_decision.get("decision", "HOLD")
        confidence = self.final_decision.get("confidence", 0)
        
        dec_color = self._get_signal_color(decision)
        
        content = Text()
        content.append("\n", style="")
        content.append(f"  {decision}  ", style=f"bold white on {dec_color}")
        content.append(f"\n\n  Confidence: {confidence}%", style=COLORS["muted"])
        
        if self.requires_review:
            content.append("\n\n  ", style="")
            content.append("[!] HUMAN REVIEW REQUIRED", style=f"bold {COLORS['warning']}")
            if self.review_triggers:
                content.append(f"\n  Triggers: {', '.join(self.review_triggers)}", style=f"dim {COLORS['warning']}")
        
        return Panel(
            Align.center(content),
            title="[bold]Final Decision[/bold]",
            title_align="left",
            border_style=dec_color,
            padding=(1, 2),
        )
    
    def create_messages_panel(self) -> Optional[Panel]:
        """Create recent messages panel."""
        if not self.messages:
            return None
        
        content = Text()
        for msg_type, msg in self.messages[-5:]:
            if msg_type == "info":
                content.append(":: ", style=f"bold {COLORS['primary']}")
            elif msg_type == "success":
                content.append(">> ", style=f"bold {COLORS['accent']}")
            elif msg_type == "warning":
                content.append("** ", style=f"bold {COLORS['warning']}")
            elif msg_type == "error":
                content.append("!! ", style=f"bold {COLORS['danger']}")
            else:
                content.append("   ", style="")
            
            content.append(msg + "\n", style=COLORS["muted"])
        
        return Panel(
            content,
            title="[bold]Activity Log[/bold]",
            title_align="left",
            border_style=COLORS["muted"],
            padding=(0, 1),
        )
    
    def create_display(self) -> Group:
        """Create the full display."""
        self.frame_idx += 1
        
        elements = [
            self.create_header(),
            self.create_agents_panel(),
        ]
        
        debate_panel = self.create_debate_panel()
        if debate_panel:
            elements.append(debate_panel)
        
        decision_panel = self.create_decision_panel()
        if decision_panel:
            elements.append(decision_panel)
        
        messages_panel = self.create_messages_panel()
        if messages_panel:
            elements.append(messages_panel)
        
        return Group(*elements)
    
    def handle_event(self, event: dict[str, Any]):
        """Handle a streaming event and update display state."""
        event_type = event.get("type", "")
        
        if event_type == "agent_start":
            agent = event.get("agent", "")
            if agent in self.agents:
                self.agents[agent].status = "running"
                self.agents[agent].start_time = time.time()
                self.add_message("info", f"{agent.replace('_', ' ').title()} started")
        
        elif event_type == "agent_thinking":
            agent = event.get("agent", "")
            content = event.get("content", "")
            if agent in self.agents:
                self.agents[agent].thinking = content
        
        elif event_type == "agent_done":
            agent = event.get("agent", "")
            if agent in self.agents:
                self.agents[agent].status = "done"
                self.agents[agent].thinking = ""
                self.agents[agent].end_time = time.time()
                
                signal = event.get("signal")
                if signal:
                    self.agents[agent].signal = signal
                    conf = event.get("confidence", 0)
                    if isinstance(conf, float) and conf <= 1:
                        conf = int(conf * 100)
                    self.agents[agent].confidence = conf
                
                summary = event.get("summary", event.get("reason", ""))
                self.agents[agent].summary = summary
                
                self.add_message("success", f"{agent.replace('_', ' ').title()} complete")
        
        elif event_type == "debate_start":
            self.debate.active = True
            self.debate.positions = event.get("positions", {})
            self.add_message("info", "Consensus debate started")
        
        elif event_type == "debate_round":
            self.debate.round_num = event.get("round", 0)
        
        elif event_type == "agent_argument":
            self.debate.arguments.append({
                "agent": event.get("agent", ""),
                "position": event.get("position", ""),
                "message": event.get("message", ""),
            })
        
        elif event_type == "position_update":
            agent = event.get("agent", "")
            new_pos = event.get("new_position", "")
            if agent and new_pos:
                self.debate.positions[agent] = new_pos
                self.add_message("warning", f"{agent} changed position to {new_pos}")
        
        elif event_type == "consensus_result":
            self.debate.result = event.get("decision", "")
            reached = event.get("consensus_reached", False)
            if reached:
                self.add_message("success", f"Consensus reached: {self.debate.result}")
            else:
                self.add_message("info", f"Majority decision: {self.debate.result}")
        
        elif event_type == "human_review":
            self.requires_review = True
            self.review_triggers = event.get("triggers", [])
            self.add_message("warning", "Human review required")
        
        elif event_type == "final_decision":
            self.final_decision = {
                "decision": event.get("decision", "HOLD"),
                "confidence": event.get("confidence", 0),
            }
            self.requires_review = event.get("requires_review", False)
            self.review_triggers = event.get("review_triggers", [])
    
    def add_message(self, msg_type: str, message: str):
        """Add a message to the log."""
        self.messages.append((msg_type, message))
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]


class CompactAgentDisplay:
    """Compact single-line agent display for simpler output."""
    
    def __init__(self, console: Console):
        self.console = console
        self.frame_idx = 0
    
    def print_agent_start(self, agent: str):
        """Print agent start message."""
        color, icon = get_agent_style(agent)
        name = agent.replace("_", " ").title()
        
        text = Text()
        text.append(f"{icon} ", style=f"bold {color}")
        text.append(name, style=f"bold {color}")
        text.append(" ", style="")
        
        frames = SPINNER_FRAMES["dots"]
        frame = frames[0]
        text.append(frame, style=f"{color}")
        
        self.console.print(text)
    
    def print_agent_thinking(self, agent: str, content: str):
        """Print agent thinking."""
        color, _ = get_agent_style(agent)
        
        display = content[:70] + "..." if len(content) > 70 else content
        text = Text()
        text.append("   >> ", style=f"dim {color}")
        text.append(display, style=f"italic {COLORS['muted']}")
        
        self.console.print(text)
    
    def print_agent_done(self, agent: str, signal: str = None, confidence: int = None):
        """Print agent completion."""
        color, icon = get_agent_style(agent)
        name = agent.replace("_", " ").title()
        
        text = Text()
        text.append(f"{icon} ", style=f"bold {COLORS['accent']}")
        text.append(name, style=f"{COLORS['accent']}")
        text.append(" >> Complete", style=f"dim {COLORS['accent']}")
        
        if signal:
            sig_color = COLORS["accent"] if "bull" in signal.lower() or "buy" in signal.lower() else (
                COLORS["danger"] if "bear" in signal.lower() or "sell" in signal.lower() else COLORS["warning"]
            )
            text.append(f" | {signal}", style=f"bold {sig_color}")
            if confidence:
                text.append(f" ({confidence}%)", style=f"dim {sig_color}")
        
        self.console.print(text)
    
    def print_debate_start(self, positions: dict):
        """Print debate start."""
        self.console.print()
        text = Text()
        text.append("<> ", style=f"bold {COLORS['secondary']}")
        text.append("CONSENSUS DEBATE", style=f"bold {COLORS['secondary']}")
        self.console.print(text)
        
        for agent, pos in positions.items():
            pos_color = COLORS["accent"] if "bull" in pos.lower() else (
                COLORS["danger"] if "bear" in pos.lower() else COLORS["warning"]
            )
            self.console.print(f"   {agent}: ", style=COLORS["muted"], end="")
            self.console.print(pos, style=f"bold {pos_color}")
    
    def print_consensus(self, decision: str, reached: bool):
        """Print consensus result."""
        text = Text()
        text.append("<> ", style=f"bold {COLORS['accent']}")
        if reached:
            text.append("CONSENSUS: ", style=f"bold {COLORS['accent']}")
        else:
            text.append("MAJORITY: ", style=f"bold {COLORS['warning']}")
        text.append(decision, style=f"bold {COLORS['primary']}")
        self.console.print(text)
        self.console.print()
    
    def print_final_decision(self, decision: str, confidence: int, requires_review: bool = False):
        """Print final decision."""
        dec_color = COLORS["accent"] if "buy" in decision.lower() else (
            COLORS["danger"] if "sell" in decision.lower() else COLORS["warning"]
        )
        
        self.console.print()
        self.console.print("=" * 60, style=COLORS["primary"])
        
        text = Text()
        text.append("  FINAL DECISION: ", style=f"bold {COLORS['muted']}")
        text.append(f" {decision} ", style=f"bold white on {dec_color}")
        text.append(f"  ({confidence}% confidence)", style=COLORS["muted"])
        self.console.print(text)
        
        if requires_review:
            self.console.print("  ** Human Review Required", style=f"bold {COLORS['warning']}")
        
        self.console.print("=" * 60, style=COLORS["primary"])
        self.console.print()
