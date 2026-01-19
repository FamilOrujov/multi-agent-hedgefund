
import time
from typing import Optional, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.box import ROUNDED, HEAVY

from src.cli.branding import COLORS, create_section_header
from src.cli.animations import print_success, print_error, print_warning, print_info


class HITLPrompt:
    """Interactive HITL prompts with rich formatting."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def create_review_panel(self, review_data: dict) -> Panel:
        """Create a styled review request panel."""
        content = Text()
        
        content.append("REVIEW ID: ", style=f"bold {COLORS['muted']}")
        content.append(f"{review_data.get('review_id', 'N/A')}\n", style=COLORS["primary"])
        
        content.append("TICKER: ", style=f"bold {COLORS['muted']}")
        content.append(f"{review_data.get('ticker', 'N/A')}\n\n", style=f"bold {COLORS['primary']}")
        
        decision = review_data.get("decision", "HOLD")
        dec_color = self._get_decision_color(decision)
        content.append("PROPOSED DECISION: ", style=f"bold {COLORS['muted']}")
        content.append(f" {decision} ", style=f"bold white on {dec_color}")
        content.append("\n")
        
        confidence = review_data.get("confidence", 0)
        content.append("CONFIDENCE: ", style=f"bold {COLORS['muted']}")
        content.append(f"{confidence}%\n", style=COLORS["primary"])
        
        position_size = review_data.get("position_size", "None")
        content.append("POSITION SIZE: ", style=f"bold {COLORS['muted']}")
        content.append(f"{position_size}\n", style=COLORS["primary"])
        
        return Panel(
            content,
            title="[bold]Human Review Required[/bold]",
            title_align="left",
            border_style=COLORS["warning"],
            padding=(1, 2),
        )
    
    def create_triggers_panel(self, triggers: list) -> Panel:
        """Create panel showing review triggers."""
        trigger_messages = {
            "low_confidence": "Low confidence in the decision",
            "conflicting_signals": "Conflicting signals between analysts",
            "high_risk": "High-risk decision type",
            "large_position": "Large position size recommended",
            "volatile_market": "High market volatility detected",
            "data_quality": "Data quality concerns",
        }
        
        content = Text()
        for trigger in triggers:
            msg = trigger_messages.get(trigger, trigger)
            content.append("** ", style=f"bold {COLORS['warning']}")
            content.append(f"{msg}\n", style=COLORS["warning"])
        
        return Panel(
            content,
            title="[bold]Review Triggers[/bold]",
            title_align="left",
            border_style=COLORS["warning"],
            padding=(0, 1),
        )
    
    def create_signals_table(self, signals: dict) -> Table:
        """Create table showing agent signals."""
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['secondary']}",
            border_style=COLORS["muted"],
            box=ROUNDED,
            expand=True,
        )
        
        table.add_column("Agent", style="bold")
        table.add_column("Signal", justify="center")
        table.add_column("Confidence", justify="center")
        
        for agent, signal_data in signals.items():
            if not isinstance(signal_data, dict):
                continue
            
            signal = signal_data.get("signal", "N/A")
            conf = signal_data.get("confidence", 0)
            
            if isinstance(conf, float) and conf <= 1:
                conf = int(conf * 100)
            
            sig_color = self._get_decision_color(signal)
            
            table.add_row(
                Text(agent.title(), style=COLORS["primary"]),
                Text(signal, style=f"bold {sig_color}"),
                Text(f"{conf}%", style=COLORS["muted"]),
            )
        
        return table
    
    def _get_decision_color(self, decision: str) -> str:
        """Get color for decision/signal."""
        dec_lower = decision.lower()
        if "bull" in dec_lower or "buy" in dec_lower:
            return COLORS["accent"]
        elif "bear" in dec_lower or "sell" in dec_lower:
            return COLORS["danger"]
        return COLORS["warning"]
    
    def display_review_request(self, review_data: dict):
        """Display the full review request."""
        self.console.print()
        self.console.print("=" * 60, style=COLORS["warning"])
        self.console.print()
        
        self.console.print(self.create_review_panel(review_data))
        
        triggers = review_data.get("triggers", [])
        if triggers:
            self.console.print(self.create_triggers_panel(triggers))
        
        signals = review_data.get("signals", {})
        if signals:
            self.console.print()
            self.console.print("Agent Signals:", style=f"bold {COLORS['secondary']}")
            self.console.print(self.create_signals_table(signals))
        
        thesis = review_data.get("thesis_summary", "")
        if thesis:
            self.console.print()

            from rich.console import Group
            from rich.markdown import Markdown

            try:
                sections = []
                buffer: list[str] = []

                for line in thesis.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("## "):
                        if buffer:
                            sections.append(Markdown("\n".join(buffer).strip()))
                            buffer = []
                        heading = Text(stripped[3:].strip(), style=f"bold {COLORS['secondary']}")
                        sections.append(Align.center(heading))
                    else:
                        buffer.append(line)

                if buffer:
                    sections.append(Markdown("\n".join(buffer).strip()))

                thesis_content = Group(*sections) if sections else Markdown(thesis)
            except Exception:
                thesis_content = Text(thesis, style=f"italic {COLORS['muted']}")

            self.console.print(Panel(
                thesis_content,
                title="[bold]Thesis Summary[/bold]",
                title_align="center",
                border_style=COLORS["muted"],
                padding=(1, 2),
                expand=False,
            ))
        
        self.console.print()
    
    def get_review_decision(self, review_data: dict) -> dict:
        """
        Interactive prompt for review decision.
        
        Returns dict with:
            - approved: bool
            - modified_decision: Optional[str]
            - notes: str
            - quit: bool (if user wants to quit)
        """
        self.display_review_request(review_data)
        
        options_text = Text()
        options_text.append("\nOPTIONS:\n", style=f"bold {COLORS['primary']}")
        options_text.append("  [A] ", style=f"bold {COLORS['accent']}")
        options_text.append("Approve - Accept the proposed decision\n", style=COLORS["muted"])
        options_text.append("  [R] ", style=f"bold {COLORS['danger']}")
        options_text.append("Reject  - Reject and halt analysis\n", style=COLORS["muted"])
        options_text.append("  [E] ", style=f"bold {COLORS['warning']}")
        options_text.append("Edit    - Modify the decision\n", style=COLORS["muted"])
        options_text.append("  [Q] ", style=f"bold {COLORS['muted']}")
        options_text.append("Quit    - Exit without decision\n", style=COLORS["muted"])
        
        self.console.print(options_text)
        self.console.print("-" * 60, style=COLORS["muted"])
        
        while True:
            choice = Prompt.ask(
                Text("Your choice", style=f"bold {COLORS['primary']}"),
                choices=["a", "r", "e", "q", "A", "R", "E", "Q"],
                default="a",
                console=self.console,
            ).upper()
            
            if choice == "A":
                notes = Prompt.ask(
                    Text("Notes (optional)", style=COLORS["muted"]),
                    default="",
                    console=self.console,
                )
                if notes:
                    self.console.print(f"\n  📝 Note recorded: \"{notes[:50]}{'...' if len(notes) > 50 else ''}\"", style=f"italic {COLORS['accent']}")
                print_success(self.console, "Decision approved!")
                return {
                    "approved": True,
                    "modified_decision": None,
                    "notes": notes or "Approved by human reviewer",
                    "quit": False,
                }
            
            elif choice == "R":
                reason = Prompt.ask(
                    Text("Reason for rejection", style=COLORS["muted"]),
                    default="",
                    console=self.console,
                )
                print_error(self.console, "Decision rejected")
                return {
                    "approved": False,
                    "modified_decision": None,
                    "notes": reason or "Rejected by human reviewer",
                    "quit": False,
                }
            
            elif choice == "E":
                self.console.print("\n[bold]Edit Options:[/bold]", style=COLORS['primary'])
                self.console.print("  [1] Change decision (BUY/SELL/HOLD)", style=COLORS["muted"])
                self.console.print("  [2] Add instructions for re-analysis", style=COLORS["muted"])
                self.console.print("  [3] Adjust confidence threshold", style=COLORS["muted"])
                
                edit_choice = Prompt.ask(
                    Text("Edit option", style=f"bold {COLORS['primary']}"),
                    choices=["1", "2", "3"],
                    default="1",
                    console=self.console,
                )
                
                if edit_choice == "1":
                    self.console.print("\nAvailable decisions:", style=f"bold {COLORS['muted']}")
                    decisions = ["BUY", "SELL", "HOLD", "STRONG_BUY", "STRONG_SELL"]
                    for i, dec in enumerate(decisions, 1):
                        dec_color = self._get_decision_color(dec)
                        self.console.print(f"  [{i}] {dec}", style=dec_color)
                    
                    new_decision = Prompt.ask(
                        Text("New decision", style=f"bold {COLORS['primary']}"),
                        choices=["1", "2", "3", "4", "5"] + decisions,
                        console=self.console,
                    ).upper()
                    
                    if new_decision.isdigit():
                        idx = int(new_decision) - 1
                        if 0 <= idx < len(decisions):
                            new_decision = decisions[idx]
                    
                    if new_decision not in decisions:
                        print_error(self.console, "Invalid decision. Please try again.")
                        continue
                    
                    notes = Prompt.ask(
                        Text("Reason for change", style=COLORS["muted"]),
                        default="",
                        console=self.console,
                    )
                    
                    if notes:
                        self.console.print(f"\n  📝 Note recorded: \"{notes[:50]}{'...' if len(notes) > 50 else ''}\"", style=f"italic {COLORS['accent']}")
                    
                    print_warning(self.console, f"Decision modified to: {new_decision}")
                    return {
                        "approved": True,
                        "modified_decision": new_decision,
                        "notes": notes or f"Modified to {new_decision} by human reviewer",
                        "quit": False,
                    }
                
                elif edit_choice == "2":
                    self.console.print("\n[bold]Re-analysis Instructions[/bold]", style=COLORS['warning'])
                    self.console.print("  Tell the agents what to focus on or reconsider:", style=f"dim {COLORS['muted']}")
                    
                    instructions = Prompt.ask(
                        Text("Your instructions", style=f"bold {COLORS['primary']}"),
                        console=self.console,
                    )
                    
                    if instructions:
                        self.console.print(f"\n  📝 Instructions recorded: \"{instructions[:60]}{'...' if len(instructions) > 60 else ''}\"", style=f"italic {COLORS['accent']}")
                        self.console.print("  ⟳ Agents will consider this in future analysis", style=f"dim {COLORS['muted']}")
                    
                    print_info(self.console, "Instructions saved - approve to continue with current decision")
                    return {
                        "approved": True,
                        "modified_decision": None,
                        "notes": f"Re-analysis requested: {instructions}",
                        "reanalysis_instructions": instructions,
                        "quit": False,
                    }
                
                elif edit_choice == "3":
                    current_conf = review_data.get("confidence", 0)
                    self.console.print(f"\n  Current confidence: {current_conf}%", style=COLORS['muted'])
                    
                    new_conf = Prompt.ask(
                        Text("New confidence threshold (0-100)", style=f"bold {COLORS['primary']}"),
                        default=str(current_conf),
                        console=self.console,
                    )
                    
                    try:
                        conf_val = int(new_conf)
                        if 0 <= conf_val <= 100:
                            self.console.print(f"\n  📝 Confidence adjusted: {current_conf}% → {conf_val}%", style=f"italic {COLORS['accent']}")
                            return {
                                "approved": True,
                                "modified_decision": None,
                                "modified_confidence": conf_val,
                                "notes": f"Confidence adjusted from {current_conf}% to {conf_val}%",
                                "quit": False,
                            }
                    except ValueError:
                        pass
                    
                    print_error(self.console, "Invalid confidence value")
                    continue
            
            elif choice == "Q":
                if Confirm.ask(
                    Text("Are you sure you want to quit?", style=COLORS["warning"]),
                    default=False,
                    console=self.console,
                ):
                    return {
                        "approved": False,
                        "modified_decision": None,
                        "notes": "User quit the review process",
                        "quit": True,
                    }


class InteractiveChat:
    """Interactive chat mode for talking to the multi-agent system."""
    
    def __init__(self, console: Console):
        self.console = console
        self.history: list[tuple[str, str]] = []
    
    def display_welcome(self):
        """Display chat welcome message."""
        self.console.print()
        self.console.print(Panel(
            Align.center(Text.from_markup(
                "[bold]Interactive Chat Mode[/bold]\n\n"
                "Ask questions about stocks, get analysis insights,\n"
                "or discuss investment strategies with the AI agents.\n\n"
                "[dim]Type 'exit' or 'quit' to leave chat mode[/dim]"
            )),
            border_style=COLORS["primary"],
            padding=(1, 2),
        ))
        self.console.print()
    
    def get_input(self) -> Optional[str]:
        """Get user input with styled prompt."""
        try:
            user_input = Prompt.ask(
                Text("You", style=f"bold {COLORS['accent']}"),
                console=self.console,
            )
            
            if user_input.lower() in ["exit", "quit", "q"]:
                return None
            
            return user_input
        except (KeyboardInterrupt, EOFError):
            return None
    
    def display_response(self, agent: str, response: str, streaming: bool = True):
        """Display agent response."""
        color, icon = self._get_agent_style(agent)
        
        header = Text()
        header.append(f"{icon} ", style=f"bold {color}")
        header.append(agent, style=f"bold {color}")
        self.console.print(header)
        
        if streaming:
            for i, char in enumerate(response):
                self.console.print(char, end="", style=COLORS["muted"])
                if i % 3 == 0:
                    time.sleep(0.01)
            self.console.print()
        else:
            self.console.print(response, style=COLORS["muted"])
        
        self.console.print()
        self.history.append(("user", ""))
        self.history.append((agent, response))
    
    def _get_agent_style(self, agent: str) -> tuple[str, str]:
        """Get styling for agent."""
        agent_lower = agent.lower()
        if "technical" in agent_lower:
            return COLORS["secondary"], "##"
        elif "fundamental" in agent_lower:
            return COLORS["accent"], "$$"
        elif "sentiment" in agent_lower:
            return COLORS["warning"], "@@"
        elif "portfolio" in agent_lower:
            return COLORS["danger"], "**"
        else:
            return COLORS["primary"], ">>"


def format_analysis_summary(result: dict, console: Console):
    """Format and display analysis summary."""
    console.print()
    console.print(create_section_header("ANALYSIS SUMMARY"))
    console.print()
    
    ticker = result.get("ticker", "N/A")
    decision = result.get("manager_decision", {})
    
    summary_table = Table(
        show_header=False,
        border_style=COLORS["muted"],
        box=ROUNDED,
        expand=True,
        padding=(0, 2),
    )
    
    summary_table.add_column("Field", style=f"bold {COLORS['muted']}")
    summary_table.add_column("Value", style=COLORS["primary"])
    
    summary_table.add_row("Ticker", ticker)
    
    dec_text = decision.get("decision", "HOLD")
    dec_color = COLORS["accent"] if "buy" in dec_text.lower() else (
        COLORS["danger"] if "sell" in dec_text.lower() else COLORS["warning"]
    )
    summary_table.add_row("Decision", Text(dec_text, style=f"bold {dec_color}"))
    
    summary_table.add_row("Confidence", f"{decision.get('confidence', 0)}%")
    summary_table.add_row("Position Size", decision.get("position_size", "N/A"))
    
    human_review = result.get("human_review", {})
    if human_review:
        summary_table.add_row("Human Reviewed", "Yes")
        if human_review.get("modified_decision"):
            summary_table.add_row("Modified To", human_review["modified_decision"])
        if human_review.get("notes"):
            summary_table.add_row("Notes", human_review["notes"][:50])
    
    console.print(summary_table)
    console.print()
