
import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Callable, Any
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.shortcuts import CompleteStyle

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.align import Align
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.box import ROUNDED, HEAVY, DOUBLE


# ASCII art logo with gradient colors (top: purple, bottom: gold/amber)
# Each tuple contains: (line_text, color)
LOGO_GRADIENT = [
    (" █████╗ ███████╗ ██████╗ ██╗███████╗    ███████╗██╗     ██╗   ██╗██╗  ██╗", "#a855f7"),  # Purple
    ("██╔══██╗██╔════╝██╔════╝ ██║██╔════╝    ██╔════╝██║     ██║   ██║╚██╗██╔╝", "#b668f7"),  # Light purple
    ("███████║█████╗  ██║  ███╗██║███████╗    █████╗  ██║     ██║   ██║ ╚███╔╝ ", "#c98b3a"),  # Transition
    ("██╔══██║██╔══╝  ██║   ██║██║╚════██║    ██╔══╝  ██║     ██║   ██║ ██╔██╗ ", "#e8a832"),  # Gold
    ("██║  ██║███████╗╚██████╔╝██║███████║    ██║     ███████╗╚██████╔╝██╔╝ ██╗", "#f5b523"),  # Light gold
    ("╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝    ╚═╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝", "#fbbf24"),  # Bright gold
]

# Subtitle - plain text for better terminal compatibility
SUBTITLE_TEXT = "Multi-Agent Investment System"


VERSION = "v0.1.0"

COLORS = {
    "primary": "#a855f7",      # Purple
    "secondary": "#fbbf24",    # Yellow/Gold
    "accent": "#22c55e",       # Green
    "warning": "#f59e0b",      # Orange
    "danger": "#ef4444",       # Red
    "muted": "#9ca3af",        # Gray
    "purple": "#c084fc",       # Light purple
    "yellow": "#facc15",       # Bright yellow
    "green": "#4ade80",        # Light green
    "aegis_blue": "#0096FF",   # Vivid Blue
    "flux_gray": "#D4D4D8",    # Bright Gray
}

PROMPT_STYLE = Style.from_dict({
    "prompt": "#a855f7 bold",
    "command": "#fbbf24",
    "path": "#9ca3af",
})


class CommandCompleter(Completer):
    """Custom completer for slash commands."""
    
    def __init__(self, commands: dict):
        self.commands = commands
    
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        
        if text.startswith("/"):
            cmd_text = text[1:]
            for cmd, info in self.commands.items():
                if cmd.startswith(cmd_text):
                    yield Completion(
                        cmd,
                        start_position=-len(cmd_text),
                        display=HTML(f"<b>/{cmd}</b>"),
                        display_meta=info.get("description", ""),
                    )
        
        elif text.startswith("/analyze ") or text.startswith("/a "):
            prefix = "/analyze " if text.startswith("/analyze ") else "/a "
            ticker_part = text[len(prefix):]
            tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AMD", "NFLX", "DIS"]
            for ticker in tickers:
                if ticker.lower().startswith(ticker_part.lower()):
                    yield Completion(
                        ticker,
                        start_position=-len(ticker_part),
                        display=ticker,
                    )


class AegisFluxCLI:
    """Interactive CLI for Aegis Flux - Claude Code style."""
    
    def __init__(self):
        self.console = Console()
        self.running = True
        self.current_mode = "chat"
        self.history: List[tuple] = []
        self.session_start = datetime.now()
        
        self.commands = {
            "help": {"handler": self.cmd_help, "description": "Show all available commands", "alias": ["h", "?"]},
            "analyze": {"handler": self.cmd_analyze, "description": "Analyze a stock ticker", "alias": ["a"]},
            "quick": {"handler": self.cmd_quick, "description": "Quick price check for a ticker", "alias": ["q"]},
            "doctor": {"handler": self.cmd_doctor, "description": "Run system diagnostics", "alias": ["d"]},
            "agents": {"handler": self.cmd_agents, "description": "View and manage agents", "alias": []},
            "config": {"handler": self.cmd_config, "description": "View/edit configuration", "alias": ["c"]},
            "status": {"handler": self.cmd_status, "description": "Show system status", "alias": ["s"]},
            "reports": {"handler": self.cmd_reports, "description": "List analysis reports", "alias": ["r"]},
            "watchlist": {"handler": self.cmd_watchlist, "description": "Manage your watchlist", "alias": ["w"]},
            "indicators": {"handler": self.cmd_indicators, "description": "Show technical indicators for a ticker", "alias": ["i"]},
            "compare": {"handler": self.cmd_compare, "description": "Compare multiple tickers", "alias": []},
            "history": {"handler": self.cmd_history, "description": "Show command history", "alias": []},
            "logs": {"handler": self.cmd_logs, "description": "View and manage HITL instructions", "alias": ["l"]},
            "clear": {"handler": self.cmd_clear, "description": "Clear the screen", "alias": ["cls"]},
            "model": {"handler": self.cmd_model, "description": "Change LLM model", "alias": ["m"]},
            "models": {"handler": self.cmd_models, "description": "List available LLM models", "alias": []},
            "chat": {"handler": self.cmd_chat, "description": "Chat with AI agents", "alias": []},
            "compact": {"handler": self.cmd_compact, "description": "Toggle compact mode", "alias": []},
            "debug": {"handler": self.cmd_debug, "description": "Toggle debug mode", "alias": []},
            "export": {"handler": self.cmd_export, "description": "Export last analysis", "alias": ["e"]},
            "version": {"handler": self.cmd_version, "description": "Show version info", "alias": ["v"]},
            "quit": {"handler": self.cmd_quit, "description": "Exit Aegis Flux", "alias": ["exit"]},
        }
        
        self.completer = CommandCompleter(self.commands)
        
        history_path = Path.home() / ".aegisflux_history"
        self.session = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            style=PROMPT_STYLE,
            complete_while_typing=True,
            mouse_support=False,  # Disabled to allow terminal scrolling
            complete_style=CompleteStyle.COLUMN,
        )
        
        self.debug_mode = False
        self.compact_mode = False
        self.current_model = "llama3.2:3b"
        self.current_embedding_model = "nomic-embed-text:latest"
        self.last_result = None
        self.analysis_task = None
    
    def render_welcome(self):
        """Render the welcome screen."""
        # Clear terminal using OS command for reliable clear
        os.system('clear' if os.name != 'nt' else 'cls')
        
        # Get terminal width for centering
        term_width = self.console.width or 80
        
        # Calculate max logo line width for centering
        max_logo_width = max(len(line) for line, _ in LOGO_GRADIENT)
        
        # Print logo centered with gradient colors
        print()
        for line, color in LOGO_GRADIENT:
            padding = (term_width - max_logo_width) // 2
            self.console.print(
                " " * padding + f"[bold {color}]{line}[/]"
            )
        
        # Print subtitle in purple
        print()
        padding = (term_width - len(SUBTITLE_TEXT)) // 2
        self.console.print(" " * padding + f"[italic #c084fc]{SUBTITLE_TEXT}[/]")

        
        print()
        padding = (term_width - len(VERSION)) // 2
        self.console.print(" " * padding + f"[dim #9ca3af]{VERSION}[/]")
        
        print()
        model_info = f"Model: {self.current_model}  Type /help for commands"
        padding = (term_width - len(model_info)) // 2
        self.console.print(
            " " * padding + f"[bold #fbbf24]Model: {self.current_model}[/] "
            f"[#9ca3af]Type[/] [bold #a855f7]/help[/] [#9ca3af]for commands[/]"
        )
        print()


    
    def get_prompt(self) -> HTML:
        """Get the prompt HTML."""
        return HTML(f'<prompt>❯ </prompt>')
    
    def get_bottom_toolbar(self) -> HTML:
        """Get bottom toolbar content."""
        return HTML(
            f'<b>?</b> for shortcuts  '
            f'<style bg="#333333"> {self.current_model} </style>  '
            f'<style fg="#6b7280">Press / for commands</style>'
        )
    
    async def process_input(self, user_input: str):
        """Process user input - command or chat."""
        user_input = user_input.strip()
        
        if not user_input:
            return
        
        self.history.append((datetime.now(), user_input))
        
        if user_input.startswith("/"):
            await self.process_command(user_input[1:])
        elif user_input == "?":
            await self.cmd_help("")
        elif user_input.startswith("@"):
            await self.chat_with_specific_agent(user_input)
        else:
            ticker = await self._detect_ticker_intent(user_input)
            if ticker:
                self.print_info(f"Detected ticker: {ticker}")
                await self.cmd_analyze(ticker)
            else:
                await self.chat_with_agent(user_input)
    
    async def process_command(self, cmd_line: str):
        """Process a slash command."""
        parts = cmd_line.split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        for name, info in self.commands.items():
            if cmd_name == name or cmd_name in info.get("alias", []):
                handler = info["handler"]
                if asyncio.iscoroutinefunction(handler):
                    await handler(args)
                else:
                    handler(args)
                return
        
        self.print_error(f"Unknown command: /{cmd_name}. Type /help for available commands.")
    
    async def chat_with_agent(self, message: str):
        """Chat with the AI agent."""
        self.print_thinking("Thinking...")
        
        try:
            from src.llm.ollama_client import OllamaClient
            from langchain_core.messages import HumanMessage, SystemMessage
            
            client = OllamaClient()
            llm = client.get_llm(model=self.current_model, temperature=0.7)
            
            system_prompt = """You are Aegis Flux, an AI investment analyst assistant. You help users with:
- Stock analysis and market insights
- Explaining technical, fundamental, and sentiment analysis
- Investment strategies and portfolio management concepts
- Using the Aegis Flux multi-agent system

Be concise and helpful. For detailed stock analysis, suggest using /analyze <TICKER>.
Keep responses focused and under 200 words unless more detail is needed."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message),
            ]
            
            self.console.print()
            response_text = Text()
            response_text.append("◆ ", style=f"bold {COLORS['secondary']}")
            response_text.append("Aegis", style=f"bold {COLORS['secondary']}")
            self.console.print(response_text)
            
            response = await asyncio.to_thread(llm.invoke, messages)
            
            for char in response.content:
                self.console.print(char, end="", style=COLORS["muted"])
                await asyncio.sleep(0.008)
            
            self.console.print("\n")
            
        except Exception as e:
            self.print_error(f"Chat error: {e}")
    
    async def _detect_ticker_intent(self, message: str) -> Optional[str]:
        """Detect if user wants to analyze a company and extract ticker."""
        analyze_keywords = ["analyze", "analysis", "check", "look at", "research", "investigate", "what about", "how is", "tell me about"]
        message_lower = message.lower()
        
        has_intent = any(kw in message_lower for kw in analyze_keywords)
        if not has_intent:
            return None
        
        company_tickers = {
            "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "google": "GOOGL",
            "alphabet": "GOOGL", "amazon": "AMZN", "nvidia": "NVDA", "meta": "META",
            "facebook": "META", "netflix": "NFLX", "disney": "DIS", "amd": "AMD",
            "intel": "INTC", "ibm": "IBM", "oracle": "ORCL", "salesforce": "CRM",
            "adobe": "ADBE", "paypal": "PYPL", "spotify": "SPOT", "uber": "UBER",
            "lyft": "LYFT", "airbnb": "ABNB", "coinbase": "COIN", "robinhood": "HOOD",
            "palantir": "PLTR", "snowflake": "SNOW", "crowdstrike": "CRWD",
            "zoom": "ZM", "slack": "WORK", "twitter": "X", "snap": "SNAP",
            "pinterest": "PINS", "square": "SQ", "block": "SQ", "shopify": "SHOP",
            "roku": "ROKU", "unity": "U", "roblox": "RBLX", "draftkings": "DKNG",
            "berkshire": "BRK.B", "jpmorgan": "JPM", "goldman": "GS", "morgan stanley": "MS",
            "visa": "V", "mastercard": "MA", "american express": "AXP", "amex": "AXP",
            "walmart": "WMT", "costco": "COST", "target": "TGT", "home depot": "HD",
            "starbucks": "SBUX", "mcdonalds": "MCD", "nike": "NKE", "coca cola": "KO",
            "pepsi": "PEP", "boeing": "BA", "lockheed": "LMT", "spacex": "SPACE",
            "ast spacemobile": "ASTS", "asts": "ASTS", "gamestop": "GME", "amc": "AMC",
        }
        
        for company, ticker in company_tickers.items():
            if company in message_lower:
                return ticker
        
        import re
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', message)
        if ticker_match:
            potential = ticker_match.group(1)
            if potential not in ["I", "A", "THE", "AND", "OR", "IS", "IT", "TO", "OF"]:
                return potential
        
        return None
    
    async def chat_with_specific_agent(self, message: str):
        """Chat with a specific agent using @mention."""
        agents = {
            "@technical_analyst": ("Technical Analyst", COLORS["secondary"], "technical analysis, chart patterns, indicators like RSI, MACD, moving averages"),
            "@fundamental_analyst": ("Fundamental Analyst", COLORS["accent"], "financial statements, valuation ratios, earnings, balance sheets"),
            "@sentiment_analyst": ("Sentiment Analyst", COLORS["warning"], "market sentiment, news analysis, social media trends"),
            "@portfolio_manager": ("Portfolio Manager", COLORS["purple"], "investment decisions, position sizing, risk management"),
            "@data_scout": ("Data Scout", COLORS["primary"], "data gathering, market data, company information"),
        }
        
        parts = message.split(maxsplit=1)
        agent_mention = parts[0].lower()
        user_question = parts[1] if len(parts) > 1 else "What is your analysis?"
        
        agent_info = None
        for mention, info in agents.items():
            if agent_mention == mention:
                agent_info = info
                break
        
        if not agent_info:
            self.print_error(f"Unknown agent: {agent_mention}")
            self.console.print("  Available: @technical_analyst, @fundamental_analyst, @sentiment_analyst, @portfolio_manager", style=f"dim {COLORS['muted']}")
            return
        
        agent_name, color, expertise = agent_info
        self.print_thinking(f"{agent_name} is thinking...")
        
        try:
            from src.llm.ollama_client import OllamaClient
            from langchain_core.messages import HumanMessage, SystemMessage
            
            client = OllamaClient()
            llm = client.get_llm(model=self.current_model, temperature=0.7)
            
            context = ""
            if self.last_result:
                ticker = self.last_result.get("ticker", "")
                if "technical" in agent_mention:
                    context = f"\nLast analysis for {ticker}: {str(self.last_result.get('technical_analysis', ''))[:500]}"
                elif "fundamental" in agent_mention:
                    context = f"\nLast analysis for {ticker}: {str(self.last_result.get('fundamental_analysis', ''))[:500]}"
                elif "sentiment" in agent_mention:
                    context = f"\nLast analysis for {ticker}: {str(self.last_result.get('sentiment_analysis', ''))[:500]}"
                elif "portfolio" in agent_mention:
                    context = f"\nLast decision for {ticker}: {str(self.last_result.get('manager_decision', ''))[:500]}"
            
            system_prompt = f"""You are the {agent_name} in the Aegis Flux multi-agent investment system.
Your expertise: {expertise}
Respond as this specific analyst would. Be concise but insightful.
{context}"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question),
            ]
            
            self.console.print()
            header = Text()
            header.append("◆ ", style=f"bold {color}")
            header.append(agent_name, style=f"bold {color}")
            self.console.print(header)
            
            response = await asyncio.to_thread(llm.invoke, messages)
            
            for char in response.content:
                self.console.print(char, end="", style=COLORS["muted"])
                await asyncio.sleep(0.008)
            
            self.console.print("\n")
            
        except Exception as e:
            self.print_error(f"Agent error: {e}")
    
    def print_thinking(self, message: str):
        """Print thinking indicator."""
        text = Text()
        text.append("◇ ", style=f"bold {COLORS['primary']}")
        text.append(message, style=f"italic {COLORS['muted']}")
        self.console.print(text)
    
    def print_success(self, message: str):
        """Print success message."""
        text = Text()
        text.append("✓ ", style=f"bold {COLORS['accent']}")
        text.append(message, style=COLORS["accent"])
        self.console.print(text)
    
    def print_error(self, message: str):
        """Print error message."""
        text = Text()
        text.append("✗ ", style=f"bold {COLORS['danger']}")
        text.append(message, style=COLORS["danger"])
        self.console.print(text)
    
    def print_info(self, message: str):
        """Print info message."""
        text = Text()
        text.append("● ", style=f"bold {COLORS['primary']}")
        text.append(message, style=COLORS["primary"])
        self.console.print(text)
    
    def print_warning(self, message: str):
        """Print warning message."""
        text = Text()
        text.append("! ", style=f"bold {COLORS['warning']}")
        text.append(message, style=COLORS["warning"])
        self.console.print(text)

    async def cmd_help(self, args: str):
        """Show help for all commands."""
        self.console.print()
        
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"],
            box=ROUNDED,
            expand=True,
            title="[bold]Available Commands[/bold]",
            title_style=COLORS["primary"],
        )
        
        table.add_column("Command", style=f"bold {COLORS['secondary']}")
        table.add_column("Alias", style=COLORS["muted"])
        table.add_column("Description")
        
        for name, info in sorted(self.commands.items()):
            aliases = ", ".join([f"/{a}" for a in info.get("alias", [])]) or "-"
            table.add_row(f"/{name}", aliases, info["description"])
        
        self.console.print(table)
        self.console.print()
        self.console.print("  Type any message to chat with the AI agents", style=f"dim {COLORS['muted']}")
        self.console.print("  Press Tab to autocomplete commands", style=f"dim {COLORS['muted']}")
        self.console.print()

    async def cmd_analyze(self, args: str):
        """Run stock analysis."""
        ticker = args.strip().upper()
        
        if not ticker:
            self.print_error("Usage: /analyze <TICKER>  (e.g., /analyze TSLA)")
            return
        
        self.console.print()
        self.print_info(f"Starting analysis for {ticker}...")
        self.console.print()
        
        try:
            from src.graph.workflow_streaming import run_analysis_streaming
            from src.graph.guardrails import GuardrailConfig
            from src.cli.hitl import HITLPrompt
            
            config = GuardrailConfig(
                min_confidence_for_auto_approve=0.80,
                min_confidence_for_action=0.50,
                require_consensus_for_auto=True,
            )
            
            hitl_prompt = HITLPrompt(self.console)
            human_decision = None
            
            async for event in run_analysis_streaming(
                ticker,
                llm_model=self.current_model,
                guardrail_config=config,
            ):
                event_type = event.get("type", "")
                
                if event_type == "agent_start":
                    agent = event.get("agent", "")
                    self._print_agent_start(agent)
                
                elif event_type == "agent_thinking":
                    agent = event.get("agent", "")
                    content = event.get("content", "")
                    await self._stream_agent_thinking(agent, content)
                
                elif event_type == "agent_done":
                    agent = event.get("agent", "")
                    signal = event.get("signal")
                    conf = event.get("confidence", 0)
                    if isinstance(conf, float) and conf <= 1:
                        conf = int(conf * 100)
                    self._print_agent_done(agent, signal, conf)
                
                elif event_type == "debate_start":
                    positions = event.get("positions", {})
                    self._print_debate_start(positions)
                
                elif event_type == "debate_round":
                    round_num = event.get("round", 0)
                    self.console.print(f"\n   ── Round {round_num} ──", style=f"bold {COLORS['yellow']}")
                
                elif event_type == "agent_argument":
                    agent = event.get("agent", "")
                    position = event.get("position", "")
                    message = event.get("message", "")
                    await self._stream_debate_argument(agent, position, message)
                
                elif event_type == "position_update":
                    agent = event.get("agent", "")
                    old_pos = event.get("old_position", "")
                    new_pos = event.get("new_position", "")
                    self.console.print(f"   ⟳ {agent}: {old_pos} → {new_pos}", style=f"italic {COLORS['warning']}")
                
                elif event_type == "consensus_result":
                    decision = event.get("decision", "")
                    reached = event.get("consensus_reached", False)
                    self._print_consensus(decision, reached)
                
                elif event_type == "human_review":
                    review_data = self._build_review_data(event, ticker)
                    human_decision = hitl_prompt.get_review_decision(review_data)
                    
                    if human_decision.get("quit"):
                        self.print_warning("Analysis cancelled")
                        return
                    
                    # Save instruction to database if provided
                    if human_decision.get("notes") or human_decision.get("reanalysis_instructions"):
                        await self._save_instruction_to_db(
                            ticker=ticker,
                            instruction=human_decision.get("reanalysis_instructions") or human_decision.get("notes"),
                            decision=review_data.get("decision"),
                            confidence=review_data.get("confidence"),
                        )
                
                elif event_type == "final_decision":
                    decision = event.get("decision", "HOLD")
                    confidence = event.get("confidence", 0)
                    requires_review = event.get("requires_review", False)
                    self._print_final_decision(decision, confidence, requires_review)
                
                elif event_type == "complete":
                    self.last_result = event.get("result", {})
                    self.last_result["ticker"] = ticker
                    if human_decision:
                        self.last_result["human_review"] = human_decision
            
            self.print_success(f"Analysis complete for {ticker}")
            self.console.print("  Use /export to save report", style=f"dim {COLORS['muted']}")
            self.console.print()
            
        except asyncio.CancelledError:
            self.console.print()
            self.print_warning("Analysis cancelled")
        except KeyboardInterrupt:
            self.console.print()
            self.print_warning("Analysis interrupted")
        except Exception as e:
            self.print_error(f"Analysis failed: {e}")
    
    def _get_agent_style(self, agent: str) -> tuple:
        """Get icon and color for agent."""
        styles = {
            "data_scout": (">>", COLORS["primary"]),
            "technical": ("##", COLORS["secondary"]),
            "fundamental": ("$$", COLORS["accent"]),
            "sentiment": ("@@", COLORS["warning"]),
            "manager": ("**", COLORS["purple"]),
            "debate": ("<>", COLORS["yellow"]),
            "guardrails": ("[]", COLORS["green"]),
        }
        for key, (icon, color) in styles.items():
            if key in agent.lower():
                return icon, color
        return ("●", COLORS["muted"])
    
    def _print_agent_start(self, agent: str):
        """Print agent start with spinner animation."""
        icon, color = self._get_agent_style(agent)
        text = Text()
        text.append(f"{icon} ", style=f"bold {color}")
        text.append(agent, style=f"bold {color}")
        text.append(" ⠋", style=f"dim {COLORS['muted']}")
        self.console.print(text)
    
    async def _stream_agent_thinking(self, agent: str, content: str):
        """Stream agent thinking text with typing animation."""
        if not content:
            return
        
        icon, color = self._get_agent_style(agent)
        self.console.print(f"   ", end="")
        
        for char in content[:200]:
            self.console.print(char, end="", style=f"dim {COLORS['muted']}")
            await asyncio.sleep(0.008)
        
        if len(content) > 200:
            self.console.print("...", style=f"dim {COLORS['muted']}")
        else:
            self.console.print()
    
    async def _stream_debate_argument(self, agent: str, position: str, message: str):
        """Stream debate argument with typing animation."""
        agent_colors = {
            "Technical": COLORS["secondary"],
            "Fundamental": COLORS["accent"],
            "Sentiment": COLORS["warning"],
        }
        color = agent_colors.get(agent.split()[0], COLORS["muted"])
        
        self.console.print(f"\n   ◆ {agent} ({position}):", style=f"bold {color}")
        self.console.print("     ", end="")
        
        for char in message:
            self.console.print(char, end="", style=f"{COLORS['muted']}")
            await asyncio.sleep(0.008)
        
        self.console.print()
    
    def _print_agent_done(self, agent: str, signal: str, confidence: int):
        """Print agent completion."""
        icon, color = self._get_agent_style(agent)
        text = Text()
        text.append(f"{icon} ", style=f"bold {color}")
        text.append(agent, style=f"bold {color}")
        text.append(" >> ", style=COLORS["muted"])
        text.append("Complete", style=f"bold {COLORS['accent']}")
        if signal:
            text.append(f" | {signal}", style=f"bold {color}")
            text.append(f" ({confidence}%)", style=COLORS["muted"])
        self.console.print(text)
    
    def _print_debate_start(self, positions: dict):
        """Print debate start."""
        self.console.print()
        text = Text()
        text.append("<> ", style=f"bold {COLORS['yellow']}")
        text.append("CONSENSUS DEBATE", style=f"bold {COLORS['yellow']}")
        self.console.print(text)
        
        for agent, pos in positions.items():
            self.console.print(f"   {agent}: {pos}", style=COLORS["muted"])
    
    def _print_consensus(self, decision: str, reached: bool):
        """Print consensus result."""
        text = Text()
        text.append("<> ", style=f"bold {COLORS['yellow']}")
        text.append("MAJORITY: ", style=f"bold {COLORS['yellow']}")
        text.append(decision, style=f"bold {COLORS['accent'] if reached else COLORS['warning']}")
        self.console.print(text)
        self.console.print()
    
    def _print_final_decision(self, decision: str, confidence: float, requires_review: bool):
        """Print final decision with animation."""
        if isinstance(confidence, float) and confidence <= 1:
            confidence = int(confidence * 100)
        
        self.console.print()
        self.console.print("=" * 50, style=COLORS["primary"])
        
        text = Text()
        text.append("  FINAL DECISION: ", style=f"bold {COLORS['purple']}")
        
        dec_color = COLORS["accent"] if "BUY" in decision else (COLORS["danger"] if "SELL" in decision else COLORS["secondary"])
        text.append(f" {decision} ", style=f"bold reverse {dec_color}")
        text.append(f"  ({confidence}% confidence)", style=COLORS["muted"])
        
        self.console.print(text)
        
        if requires_review:
            self.console.print("  ** Human Review Required", style=f"bold {COLORS['warning']}")
        
        self.console.print("=" * 50, style=COLORS["primary"])
        self.console.print()
    
    def _build_review_data(self, event: dict, ticker: str) -> dict:
        """Build review data from event."""
        state = event.get("state", {})
        manager_dec = state.get("manager_decision", {})
        
        return {
            "review_id": event.get("review_id", f"review_{datetime.now().strftime('%H%M%S')}"),
            "ticker": ticker,
            "decision": manager_dec.get("decision", "HOLD"),
            "confidence": manager_dec.get("confidence", 0),
            "position_size": manager_dec.get("position_size", "Moderate"),
            "triggers": event.get("triggers", []),
            "signals": {
                "technical": state.get("technical_signal", {}),
                "fundamental": state.get("fundamental_signal", {}),
                "sentiment": state.get("sentiment_signal", {}),
            },
            "thesis_summary": state.get("final_thesis", ""),
        }
    
    async def _save_instruction_to_db(self, ticker: str, instruction: str, decision: str, confidence: int):
        """Save HITL instruction to PostgreSQL."""
        try:
            from src.data.instructions_db import InstructionsDB
            db = InstructionsDB()
            await db.init_tables()
            await db.add_instruction(
                instruction=instruction,
                ticker=ticker,
                decision=decision,
                confidence=confidence,
            )
            await db.disconnect()
        except Exception:
            pass  # Silently fail if DB not available

    async def cmd_doctor(self, args: str):
        """Run system diagnostics."""
        self.console.print()
        self.print_info("Running system diagnostics...")
        self.console.print()
        
        checks = [
            ("Configuration", self._check_config),
            ("Ollama Connection", self._check_ollama),
            ("LLM Model", self._check_model),
            ("Tavily API", self._check_tavily),
            ("Data Directory", self._check_data_dir),
            ("Database", self._check_database),
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Checking...", total=len(checks))
            
            for name, check_func in checks:
                progress.update(task, description=f"Checking {name}...")
                await asyncio.sleep(0.3)
                
                try:
                    status, detail = await asyncio.to_thread(check_func)
                    results.append((name, status, detail))
                except Exception as e:
                    results.append((name, "error", str(e)[:40]))
                
                progress.advance(task)
        
        self.console.print()
        
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"],
            box=ROUNDED,
        )
        
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")
        
        all_ok = True
        for name, status, detail in results:
            if status == "ok":
                status_text = Text("✓ OK", style=f"bold {COLORS['accent']}")
            elif status == "warn":
                status_text = Text("! WARN", style=f"bold {COLORS['warning']}")
                all_ok = False
            else:
                status_text = Text("✗ ERROR", style=f"bold {COLORS['danger']}")
                all_ok = False
            
            table.add_row(name, status_text, detail)
        
        self.console.print(table)
        self.console.print()
        
        if all_ok:
            self.print_success("All systems operational!")
        else:
            self.print_warning("Some issues detected. Review the results above.")
        
        self.console.print()
    
    def _check_config(self) -> tuple:
        """Check configuration."""
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            return ("ok", f"Loaded from environment")
        except Exception as e:
            return ("error", str(e)[:40])
    
    def _check_ollama(self) -> tuple:
        """Check Ollama connection."""
        try:
            import httpx
            from src.llm.ollama_client import OllamaClient
            
            client = OllamaClient()
            response = httpx.get(f"{client.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return ("ok", f"{len(models)} models available")
            return ("error", f"HTTP {response.status_code}")
        except Exception as e:
            return ("error", str(e)[:40])
    
    def _check_model(self) -> tuple:
        """Check LLM model availability."""
        try:
            import httpx
            from src.llm.ollama_client import OllamaClient
            
            client = OllamaClient()
            response = httpx.get(f"{client.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = [m.get("name", "").split(":")[0] for m in response.json().get("models", [])]
                if self.current_model.split(":")[0] in models or any(self.current_model in m for m in models):
                    return ("ok", f"{self.current_model} ready")
                return ("warn", f"{self.current_model} not found")
            return ("error", "Cannot check models")
        except Exception as e:
            return ("error", str(e)[:40])
    
    def _check_tavily(self) -> tuple:
        """Check Tavily API."""
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            
            if settings.tavily_api_key:
                return ("ok", "API key configured")
            return ("warn", "API key not set")
        except Exception:
            return ("warn", "Cannot check")
    
    def _check_data_dir(self) -> tuple:
        """Check data directory."""
        try:
            data_dir = Path("./data")
            reports_dir = data_dir / "reports"
            
            if reports_dir.exists():
                count = len(list(reports_dir.glob("*.pdf")))
                return ("ok", f"{count} reports saved")
            return ("warn", "No reports yet")
        except Exception as e:
            return ("error", str(e)[:40])
    
    def _check_database(self) -> tuple:
        """Check database (placeholder)."""
        return ("ok", "Not required for basic operation")

    async def cmd_agents(self, args: str):
        """Show agent information."""
        self.console.print()
        
        agents_info = [
            ("Data Scout", ">>", COLORS["primary"], "Collects market data, financials, and company info"),
            ("Technical Analyst", "##", COLORS["secondary"], "Analyzes price patterns and technical indicators"),
            ("Fundamental Analyst", "$$", COLORS["accent"], "Evaluates financial health and valuations"),
            ("Sentiment Analyst", "@@", COLORS["warning"], "Analyzes news sentiment and market mood"),
            ("Portfolio Manager", "**", COLORS["danger"], "Makes final investment decisions"),
        ]
        
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"],
            box=ROUNDED,
            title="[bold]Multi-Agent Team[/bold]",
            title_style=COLORS["primary"],
        )
        
        table.add_column("Icon", justify="center", width=6)
        table.add_column("Agent", style="bold")
        table.add_column("Role")
        
        for name, icon, color, role in agents_info:
            table.add_row(
                Text(icon, style=f"bold {color}"),
                Text(name, style=color),
                role,
            )
        
        self.console.print(table)
        self.console.print()

    async def cmd_config(self, args: str):
        """Show configuration."""
        self.console.print()
        
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            
            table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                border_style=COLORS["muted"],
                box=ROUNDED,
                title="[bold]Configuration[/bold]",
                title_style=COLORS["primary"],
            )
            
            table.add_column("Setting", style="bold")
            table.add_column("Value")
            
            table.add_row("Ollama URL", settings.ollama_base_url)
            table.add_row("Default Model", settings.ollama_llm_model)
            table.add_row("Current Model", self.current_model)
            table.add_row("Temperature", str(settings.agent_temperature))
            table.add_row("Data Directory", str(settings.data_dir))
            table.add_row("Tavily API", "configured" if settings.tavily_api_key else "not set")
            table.add_row("Debug Mode", "on" if self.debug_mode else "off")
            table.add_row("Compact Mode", "on" if self.compact_mode else "off")
            
            self.console.print(table)
            self.console.print()
            
        except Exception as e:
            self.print_error(f"Failed to load config: {e}")

    async def cmd_status(self, args: str):
        """Show system status."""
        self.console.print()
        
        status_text = Text()
        status_text.append("System Status\n\n", style=f"bold {COLORS['primary']}")
        status_text.append("  Session: ", style=COLORS["muted"])
        status_text.append(f"{self.session_start.strftime('%Y-%m-%d %H:%M')}\n", style=COLORS["primary"])
        status_text.append("  Model: ", style=COLORS["muted"])
        status_text.append(f"{self.current_model}\n", style=COLORS["primary"])
        status_text.append("  Commands Run: ", style=COLORS["muted"])
        status_text.append(f"{len(self.history)}\n", style=COLORS["primary"])
        status_text.append("  Last Analysis: ", style=COLORS["muted"])
        if self.last_result:
            ticker = self.last_result.get("ticker", "N/A")
            status_text.append(f"{ticker}\n", style=COLORS["primary"])
        else:
            status_text.append("None\n", style=f"dim {COLORS['muted']}")
        
        self.console.print(Panel(status_text, border_style=COLORS["muted"]))
        self.console.print()

    async def cmd_reports(self, args: str):
        """List analysis reports."""
        self.console.print()
        
        reports_dir = Path("./data/reports")
        
        if not reports_dir.exists():
            self.print_warning("No reports directory found")
            return
        
        reports = sorted(reports_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
        
        if not reports:
            self.print_info("No reports found")
            return
        
        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"],
            box=ROUNDED,
        )
        
        table.add_column("#", style="dim", width=4)
        table.add_column("Report", style="bold")
        table.add_column("Size", justify="right")
        table.add_column("Created", justify="right")
        
        for i, report in enumerate(reports, 1):
            stat = report.stat()
            size_kb = stat.st_size / 1024
            created = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            
            table.add_row(str(i), report.name, f"{size_kb:.1f} KB", created)
        
        self.console.print(table)
        self.console.print()

    async def cmd_history(self, args: str):
        """Show command history."""
        self.console.print()
        
        if not self.history:
            self.print_info("No history yet")
            return
        
        for ts, cmd in self.history[-15:]:
            time_str = ts.strftime("%H:%M:%S")
            self.console.print(f"  {time_str}  {cmd}", style=COLORS["muted"])
        
        self.console.print()

    def cmd_clear(self, args: str):
        """Clear the screen and show welcome."""
        self.render_welcome()

    async def cmd_model(self, args: str):
        """Change or show current model with 2 tables (LLMs + Embeddings)."""
        args = args.strip()
        
        if args.startswith("embed ") or args.startswith("embedding "):
            new_model = args.split(maxsplit=1)[1] if " " in args else ""
            if new_model:
                self.current_embedding_model = new_model
                self.print_success(f"Embedding model changed to: {self.current_embedding_model}")
            return
        elif args:
            self.current_model = args
            self.print_success(f"LLM model changed to: {self.current_model}")
            return
        
        self.console.print()
        
        try:
            import httpx
            from src.llm.ollama_client import OllamaClient
            
            client = OllamaClient()
            response = httpx.get(f"{client.base_url}/api/tags", timeout=10)
            
            if response.status_code != 200:
                self.print_error("Failed to fetch models from Ollama")
                return
            
            models = response.json().get("models", [])
            
            llm_models = []
            embed_models = []
            
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024 ** 3)
                
                if "embed" in name.lower() or "nomic" in name.lower() or "bge" in name.lower():
                    embed_models.append((name, size_gb))
                else:
                    llm_models.append((name, size_gb))
            
            llm_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['purple']}",
                border_style=COLORS["purple"],
                box=ROUNDED,
                title="[bold]LLM Models[/bold]",
                title_style=COLORS["purple"],
            )
            
            llm_table.add_column("Model", style="bold")
            llm_table.add_column("Size", justify="right")
            llm_table.add_column("Status", justify="center")
            
            for name, size_gb in llm_models:
                is_current = name == self.current_model
                status = Text("<- current", style=f"bold {COLORS['accent']}") if is_current else Text("")
                llm_table.add_row(name, f"{size_gb:.1f} GB", status)
            
            if not llm_models:
                llm_table.add_row("No LLM models found", "-", "")
            
            self.console.print(llm_table)
            self.console.print()
            
            embed_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['secondary']}",
                border_style=COLORS["secondary"],
                box=ROUNDED,
                title="[bold]Embedding Models[/bold]",
                title_style=COLORS["secondary"],
            )
            
            embed_table.add_column("Model", style="bold")
            embed_table.add_column("Size", justify="right")
            embed_table.add_column("Status", justify="center")
            
            for name, size_gb in embed_models:
                is_current = name == self.current_embedding_model
                status = Text("<- current", style=f"bold {COLORS['accent']}") if is_current else Text("")
                embed_table.add_row(name, f"{size_gb:.1f} GB", status)
            
            if not embed_models:
                embed_table.add_row("No embedding models found", "-", "")
            
            self.console.print(embed_table)
            self.console.print()
            
            self.console.print(f"  Current LLM: ", style=COLORS["muted"], end="")
            self.console.print(f"{self.current_model}", style=f"bold {COLORS['accent']}")
            self.console.print(f"  Current Embedding: ", style=COLORS["muted"], end="")
            self.console.print(f"{self.current_embedding_model}", style=f"bold {COLORS['accent']}")
            self.console.print()
            self.console.print("  Usage: /model <name>           - Change LLM model", style=f"dim {COLORS['muted']}")
            self.console.print("         /model embed <name>    - Change embedding model", style=f"dim {COLORS['muted']}")
            self.console.print()
            
        except Exception as e:
            self.print_error(f"Failed to fetch models: {e}")
            self.console.print(f"  Current LLM: {self.current_model}", style=COLORS["muted"])
            self.console.print(f"  Current Embedding: {self.current_embedding_model}", style=COLORS["muted"])

    async def cmd_chat(self, args: str):
        """Enter chat mode."""
        self.print_info("You're already in chat mode! Just type your message.")

    async def cmd_compact(self, args: str):
        """Toggle compact mode."""
        self.compact_mode = not self.compact_mode
        status = "enabled" if self.compact_mode else "disabled"
        self.print_success(f"Compact mode {status}")

    async def cmd_debug(self, args: str):
        """Toggle debug mode."""
        self.debug_mode = not self.debug_mode
        status = "enabled" if self.debug_mode else "disabled"
        self.print_success(f"Debug mode {status}")

    async def cmd_export(self, args: str):
        """Export last analysis to PDF."""
        if not self.last_result:
            self.print_error("No analysis to export. Run /analyze first.")
            return
        
        try:
            from src.reports.pdf_generator import PDFReportGenerator
            
            self.print_info("Generating report...")
            
            generator = PDFReportGenerator()
            ticker = self.last_result.get("ticker", "UNKNOWN")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_analysis_{timestamp}.pdf"
            
            pdf_path = generator.generate(self.last_result, filename)
            self.print_success(f"Report saved: {pdf_path}")
            
        except Exception as e:
            self.print_error(f"Export failed: {e}")

    async def cmd_quick(self, args: str):
        """Quick price check for a ticker."""
        ticker = args.strip().upper()
        
        if not ticker:
            self.print_error("Usage: /quick <TICKER>  (e.g., /quick AAPL)")
            return
        
        self.print_info(f"Fetching {ticker} data...")
        
        try:
            import yfinance as yf
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")
            
            if hist.empty:
                self.print_error(f"No data found for {ticker}")
                return
            
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = ((current - prev) / prev) * 100
            
            self.console.print()
            
            table = Table(
                show_header=False,
                border_style=COLORS["muted"],
                box=ROUNDED,
                padding=(0, 2),
            )
            
            table.add_column("Field", style=f"bold {COLORS['muted']}")
            table.add_column("Value")
            
            table.add_row("Ticker", Text(ticker, style=f"bold {COLORS['primary']}"))
            table.add_row("Name", info.get("shortName", "N/A"))
            
            price_color = COLORS["accent"] if change >= 0 else COLORS["danger"]
            table.add_row("Price", Text(f"${current:.2f}", style=f"bold {price_color}"))
            
            change_text = f"{'+' if change >= 0 else ''}{change:.2f}%"
            table.add_row("Change", Text(change_text, style=price_color))
            
            table.add_row("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")
            table.add_row("52w High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
            table.add_row("52w Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")
            
            self.console.print(table)
            self.console.print()
            
        except Exception as e:
            self.print_error(f"Failed to fetch data: {e}")

    async def cmd_watchlist(self, args: str):
        """Manage watchlist."""
        watchlist_file = Path.home() / ".aegisflux_watchlist"
        
        parts = args.strip().split()
        action = parts[0].lower() if parts else "show"
        
        if action == "add" and len(parts) > 1:
            ticker = parts[1].upper()
            current = set()
            if watchlist_file.exists():
                current = set(watchlist_file.read_text().strip().split("\n"))
            current.add(ticker)
            watchlist_file.write_text("\n".join(sorted(current)))
            self.print_success(f"Added {ticker} to watchlist")
            return
        
        elif action == "remove" and len(parts) > 1:
            ticker = parts[1].upper()
            if watchlist_file.exists():
                current = set(watchlist_file.read_text().strip().split("\n"))
                current.discard(ticker)
                watchlist_file.write_text("\n".join(sorted(current)))
                self.print_success(f"Removed {ticker} from watchlist")
            return
        
        elif action == "show" or not parts:
            if not watchlist_file.exists():
                self.print_info("Watchlist is empty. Use /watchlist add <TICKER>")
                return
            
            tickers = [t for t in watchlist_file.read_text().strip().split("\n") if t]
            
            if not tickers:
                self.print_info("Watchlist is empty")
                return
            
            self.console.print()
            self.print_info(f"Watchlist ({len(tickers)} tickers):")
            
            try:
                import yfinance as yf
                import logging
                logging.getLogger('yfinance').setLevel(logging.CRITICAL)
                
                table = Table(
                    show_header=True,
                    header_style=f"bold {COLORS['primary']}",
                    border_style=COLORS["muted"],
                    box=ROUNDED,
                )
                
                table.add_column("Ticker", style="bold")
                table.add_column("Price", justify="right")
                table.add_column("Change", justify="right")
                
                for ticker in sorted(tickers):
                    try:
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="2d")
                        if not hist.empty:
                            current = hist['Close'].iloc[-1]
                            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                            change = ((current - prev) / prev) * 100
                            change_color = COLORS["accent"] if change >= 0 else COLORS["danger"]
                            table.add_row(
                                ticker,
                                f"${current:.2f}",
                                Text(f"{'+' if change >= 0 else ''}{change:.2f}%", style=change_color),
                            )
                        else:
                            table.add_row(ticker, "-", "-")
                    except Exception:
                        table.add_row(ticker, "-", "-")
                
                self.console.print(table)
                
            except Exception:
                for ticker in sorted(tickers):
                    self.console.print(f"  {ticker}", style=COLORS["primary"])
            
            self.console.print()
            self.console.print("  Commands: /watchlist add <T> | /watchlist remove <T>", style=f"dim {COLORS['muted']}")
            self.console.print()
        else:
            self.print_error("Usage: /watchlist [add|remove] <TICKER>")

    async def cmd_indicators(self, args: str):
        """Show technical indicators for a ticker."""
        ticker = args.strip().upper()
        
        if not ticker:
            self.print_error("Usage: /indicators <TICKER>  (e.g., /indicators TSLA)")
            return
        
        self.print_info(f"Calculating indicators for {ticker}...")
        
        try:
            import yfinance as yf
            import ta
            import pandas as pd
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty or len(hist) < 20:
                self.print_error(f"Insufficient data for {ticker}")
                return
            
            rsi = ta.momentum.RSIIndicator(hist['Close']).rsi().iloc[-1]
            macd = ta.trend.MACD(hist['Close'])
            macd_val = macd.macd().iloc[-1]
            macd_sig = macd.macd_signal().iloc[-1]
            
            bb = ta.volatility.BollingerBands(hist['Close'])
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
            
            current_price = hist['Close'].iloc[-1]
            
            self.console.print()
            
            self.console.print()
            
            table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                border_style=COLORS["muted"],
                box=ROUNDED,
                title=f"[bold]Technical Indicators - {ticker}[/bold]",
                title_style=COLORS["primary"],
            )
            
            table.add_column("Indicator", style="bold")
            table.add_column("Value", justify="right")
            table.add_column("Signal")
            
            rsi_signal = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
            rsi_color = COLORS["danger"] if rsi > 70 else (COLORS["accent"] if rsi < 30 else COLORS["warning"])
            table.add_row("RSI (14)", f"{rsi:.1f}", Text(rsi_signal, style=rsi_color))
            
            macd_signal_text = "Bullish" if macd_val > macd_sig else "Bearish"
            macd_color = COLORS["accent"] if macd_val > macd_sig else COLORS["danger"]
            table.add_row("MACD", f"{macd_val:.2f}", Text(macd_signal_text, style=macd_color))
            
            bb_pos = "Upper" if current_price > bb_high else ("Lower" if current_price < bb_low else "Middle")
            table.add_row("Bollinger Band", f"{current_price:.2f}", bb_pos)
            
            table.add_row("SMA 20", f"{sma_20:.2f}", "Above" if current_price > sma_20 else "Below")
            if sma_50:
                table.add_row("SMA 50", f"{sma_50:.2f}", "Above" if current_price > sma_50 else "Below")
            
            self.console.print(table)
            self.console.print()
            
        except Exception as e:
            self.print_error(f"Failed to calculate indicators: {e}")

    async def cmd_compare(self, args: str):
        """Compare multiple tickers."""
        tickers = [t.strip().upper() for t in args.split() if t.strip()]
        
        if len(tickers) < 2:
            self.print_error("Usage: /compare <TICKER1> <TICKER2> ...  (e.g., /compare AAPL MSFT GOOGL)")
            return
        
        self.print_info(f"Comparing {', '.join(tickers)}...")
        
        try:
            import yfinance as yf
            import logging
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            
            self.console.print()
            
            table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                border_style=COLORS["muted"],
                box=ROUNDED,
                title="[bold]Stock Comparison[/bold]",
                title_style=COLORS["primary"],
            )
            
            table.add_column("Ticker", style="bold")
            table.add_column("Price", justify="right")
            table.add_column("Change", justify="right")
            table.add_column("P/E", justify="right")
            table.add_column("Mkt Cap", justify="right")
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="5d")
                    
                    if hist.empty:
                        table.add_row(ticker, "-", "-", "-", "-")
                        continue
                    
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = ((current - prev) / prev) * 100
                    
                    pe = info.get("trailingPE", 0)
                    pe_str = f"{pe:.1f}" if pe else "-"
                    
                    mkt_cap = info.get("marketCap", 0)
                    if mkt_cap >= 1e12:
                        mkt_cap_str = f"${mkt_cap/1e12:.1f}T"
                    elif mkt_cap >= 1e9:
                        mkt_cap_str = f"${mkt_cap/1e9:.1f}B"
                    else:
                        mkt_cap_str = f"${mkt_cap/1e6:.1f}M"
                    
                    change_color = COLORS["accent"] if change >= 0 else COLORS["danger"]
                    
                    table.add_row(
                        ticker,
                        f"${current:.2f}",
                        Text(f"{'+' if change >= 0 else ''}{change:.2f}%", style=change_color),
                        pe_str,
                        mkt_cap_str,
                    )
                except Exception:
                    table.add_row(ticker, "-", "-", "-", "-")
            
            self.console.print(table)
            self.console.print()
            
        except Exception as e:
            self.print_error(f"Comparison failed: {e}")

    async def cmd_models(self, args: str):
        """List available LLM models."""
        self.print_info("Fetching available models...")
        
        try:
            import httpx
            from src.llm.ollama_client import OllamaClient
            
            client = OllamaClient()
            response = httpx.get(f"{client.base_url}/api/tags", timeout=10)
            
            if response.status_code != 200:
                self.print_error("Failed to fetch models")
                return
            
            models = response.json().get("models", [])
            
            if not models:
                self.print_warning("No models found")
                return
            
            self.console.print()
            
            table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                border_style=COLORS["muted"],
                box=ROUNDED,
            )
            
            table.add_column("Model", style="bold")
            table.add_column("Size", justify="right")
            table.add_column("Active")
            
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024 ** 3)
                
                is_current = "●" if name == self.current_model else ""
                
                table.add_row(
                    name,
                    f"{size_gb:.1f} GB",
                    Text(is_current, style=f"bold {COLORS['accent']}"),
                )
            
            self.console.print(table)
            self.console.print()
            self.console.print(f"  Current: {self.current_model}", style=f"dim {COLORS['muted']}")
            self.console.print("  Use /model <name> to switch", style=f"dim {COLORS['muted']}")
            self.console.print()
            
        except Exception as e:
            self.print_error(f"Failed to fetch models: {e}")

    async def cmd_version(self, args: str):
        """Show version information."""
        self.console.print()
        
        info = Text()
        info.append("Aegis Flux ", style=f"bold {COLORS['secondary']}")
        info.append(VERSION, style=f"bold {COLORS['primary']}")
        info.append("\n\nMulti-Agent AI Investment Analysis System\n", style=COLORS["muted"])
        info.append("Powered by LangChain, LangGraph, and Ollama\n\n", style=f"dim {COLORS['muted']}")
        info.append("GitHub: ", style=COLORS["muted"])
        info.append("github.com/aegisflux\n", style=COLORS["primary"])
        
        self.console.print(Panel(info, border_style=COLORS["muted"]))
        self.console.print()

    async def cmd_logs(self, args: str):
        """View and manage HITL instructions."""
        from src.data.instructions_db import InstructionsDB
        from rich.prompt import Prompt, Confirm
        
        db = InstructionsDB()
        
        try:
            await db.init_tables()
            
            parts = args.strip().split(maxsplit=1)
            action = parts[0].lower() if parts else "list"
            
            if action == "list" or action == "":
                instructions = await db.get_all_instructions()
                
                if not instructions:
                    self.print_info("No instructions logged yet")
                    self.console.print("  Instructions are saved when you add notes during HITL review", style=f"dim {COLORS['muted']}")
                    return
                
                self.console.print()
                table = Table(
                    show_header=True,
                    header_style=f"bold {COLORS['primary']}",
                    border_style=COLORS["muted"],
                    box=ROUNDED,
                    title="[bold]HITL Instructions Log[/bold]",
                    title_style=COLORS["primary"],
                )
                
                table.add_column("ID", style="bold", width=5)
                table.add_column("Ticker", width=8)
                table.add_column("Instruction", style=COLORS["muted"])
                table.add_column("Date", width=16)
                
                for inst in instructions:
                    inst_text = inst['instruction'][:60] + "..." if len(inst['instruction']) > 60 else inst['instruction']
                    date_str = inst['created_at'].strftime("%Y-%m-%d %H:%M")
                    table.add_row(
                        str(inst['id']),
                        inst['ticker'] or "-",
                        inst_text,
                        date_str,
                    )
                
                self.console.print(table)
                self.console.print()
                self.console.print("  Commands:", style=f"bold {COLORS['muted']}")
                self.console.print("    /logs view <id>    - View full instruction", style=f"dim {COLORS['muted']}")
                self.console.print("    /logs delete <id>  - Delete instruction", style=f"dim {COLORS['muted']}")
                self.console.print("    /logs clear        - Clear all instructions", style=f"dim {COLORS['muted']}")
                self.console.print()
            
            elif action == "view":
                if len(parts) < 2:
                    self.print_error("Usage: /logs view <id>")
                    return
                
                inst_id = int(parts[1])
                instructions = await db.get_all_instructions()
                inst = next((i for i in instructions if i['id'] == inst_id), None)
                
                if not inst:
                    self.print_error(f"Instruction {inst_id} not found")
                    return
                
                self.console.print()
                content = Text()
                content.append(f"ID: ", style=f"bold {COLORS['muted']}")
                content.append(f"{inst['id']}\n", style=COLORS["primary"])
                content.append(f"Ticker: ", style=f"bold {COLORS['muted']}")
                content.append(f"{inst['ticker'] or 'N/A'}\n", style=COLORS["primary"])
                content.append(f"Date: ", style=f"bold {COLORS['muted']}")
                content.append(f"{inst['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n\n", style=COLORS["primary"])
                content.append(f"Instruction:\n", style=f"bold {COLORS['secondary']}")
                content.append(f"{inst['instruction']}\n", style=COLORS["muted"])
                
                if inst.get('decision'):
                    content.append(f"\nDecision: ", style=f"bold {COLORS['muted']}")
                    content.append(f"{inst['decision']} ", style=COLORS["accent"])
                    if inst.get('confidence'):
                        content.append(f"({inst['confidence']}%)", style=COLORS["muted"])
                
                self.console.print(Panel(content, border_style=COLORS["primary"], padding=(1, 2)))
                self.console.print()
            
            elif action == "delete":
                if len(parts) < 2:
                    self.print_error("Usage: /logs delete <id>")
                    return
                
                inst_id = int(parts[1])
                instructions = await db.get_all_instructions()
                inst = next((i for i in instructions if i['id'] == inst_id), None)
                
                if not inst:
                    self.print_error(f"Instruction {inst_id} not found")
                    return
                
                inst_preview = inst['instruction'][:50] + "..." if len(inst['instruction']) > 50 else inst['instruction']
                self.console.print(f"\n  Instruction: {inst_preview}", style=f"italic {COLORS['muted']}")
                
                if Confirm.ask(Text("Delete this instruction?", style=COLORS["warning"]), default=False, console=self.console):
                    await db.delete_instruction(inst_id)
                    self.print_success(f"Deleted instruction {inst_id}")
                else:
                    self.console.print("  Cancelled", style=f"dim {COLORS['muted']}")
            
            elif action == "clear":
                instructions = await db.get_all_instructions()
                count = len(instructions)
                
                if count == 0:
                    self.print_info("No instructions to clear")
                    return
                
                self.console.print(f"\n  This will delete {count} instruction(s)", style=COLORS["warning"])
                if Confirm.ask(Text("Clear all instructions?", style=COLORS["danger"]), default=False, console=self.console):
                    deleted = await db.clear_all_instructions()
                    self.print_success(f"Cleared {deleted} instruction(s)")
                else:
                    self.console.print("  Cancelled", style=f"dim {COLORS['muted']}")
            
            else:
                self.print_error(f"Unknown action: {action}")
                self.console.print("  Available: list, view, delete, clear", style=f"dim {COLORS['muted']}")
        
        except Exception as e:
            self.print_error(f"Database error: {e}")
            self.console.print("  Make sure PostgreSQL is running and configured", style=f"dim {COLORS['muted']}")
        finally:
            await db.disconnect()

    async def cmd_quit(self, args: str):
        """Exit the CLI."""
        self.console.print()
        self.console.print("  Goodbye! ", style=f"bold {COLORS['secondary']}", end="")
        self.console.print("👋", style="")
        self.console.print()
        self.running = False

    async def run(self):
        """Main run loop."""
        self.render_welcome()
        
        while self.running:
            try:
                user_input = await self.session.prompt_async(
                    self.get_prompt(),
                )
                
                await self.process_input(user_input)
                
            except KeyboardInterrupt:
                self.console.print()
                self.console.print("  Press Ctrl+C again to exit, or type /quit", style=f"dim {COLORS['muted']}")
                continue
            except asyncio.CancelledError:
                self.console.print()
                self.print_warning("Operation cancelled")
                continue
            except EOFError:
                await self.cmd_quit("")
                break
            except Exception as e:
                self.print_error(f"Error: {e}")


def main():
    """Entry point for interactive CLI."""
    try:
        cli = AegisFluxCLI()
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console = Console()
        console.print("\n  Goodbye! 👋\n", style="bold #fbbf24")
    except Exception as e:
        console = Console()
        console.print(f"\n  Error: {e}\n", style="bold red")


if __name__ == "__main__":
    main()


