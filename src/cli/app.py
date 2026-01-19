
import asyncio
import sys
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table

from src.cli.branding import (
    render_logo_animated,
    render_logo_static,
    render_mini_logo,
    create_header_panel,
    create_section_header,
    COLORS,
    VERSION,
)
from src.cli.animations import (
    print_success,
    print_error,
    print_warning,
    print_info,
    AgentProgressDisplay,
)
from src.cli.display import StreamingAgentDisplay, CompactAgentDisplay
from src.cli.hitl import HITLPrompt, InteractiveChat, format_analysis_summary

app = typer.Typer(
    name="aegis",
    help="Aegis Flux - Multi-Agent AI Investment Analysis System",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        render_mini_logo(console)
        console.print(f"Version: {VERSION}", style=COLORS["muted"])
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version and exit"
    ),
):
    """Aegis Flux - Multi-Agent AI Investment Analysis System."""
    pass


@app.command("analyze")
def analyze_command(
    ticker: str = typer.Argument(..., help="Stock ticker symbol to analyze"),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model to use"
    ),
    compact: bool = typer.Option(
        False, "--compact", "-c", help="Use compact display mode"
    ),
    no_animation: bool = typer.Option(
        False, "--no-animation", help="Disable animations"
    ),
    auto_approve: bool = typer.Option(
        False, "--auto-approve", "-y", help="Auto-approve HITL reviews"
    ),
    generate_report: bool = typer.Option(
        True, "--report/--no-report", "-r", help="Generate PDF report"
    ),
):
    """
    Run multi-agent analysis on a stock ticker.
    
    Examples:
        aegis analyze TSLA
        aegis analyze AAPL --model llama3.2 --compact
        aegis analyze MSFT --no-report
    """
    ticker = ticker.upper().strip()
    
    if not no_animation:
        render_logo_animated(console, speed=0.015)
    else:
        render_logo_static(console)
    
    console.print(create_header_panel(
        f"Analyzing {ticker}",
        f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ))
    console.print()
    
    try:
        result = asyncio.run(_run_analysis(
            ticker=ticker,
            model=model,
            compact=compact,
            no_animation=no_animation,
            auto_approve=auto_approve,
        ))
        
        if result is None:
            print_error(console, "Analysis cancelled")
            raise typer.Exit(1)
        
        format_analysis_summary(result, console)
        
        if generate_report:
            _generate_report(result, ticker)
        
        print_success(console, "Analysis complete!")
        
    except KeyboardInterrupt:
        console.print()
        print_warning(console, "Analysis interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        print_error(console, f"Analysis failed: {e}")
        raise typer.Exit(1)


async def _run_analysis(
    ticker: str,
    model: Optional[str],
    compact: bool,
    no_animation: bool,
    auto_approve: bool,
) -> Optional[dict]:
    """Run the analysis with streaming display."""
    from src.graph.workflow_streaming import run_analysis_streaming
    from src.graph.guardrails import GuardrailConfig
    
    config = GuardrailConfig(
        min_confidence_for_auto_approve=0.80,
        min_confidence_for_action=0.50,
        require_consensus_for_auto=True,
        always_review_decisions=["SELL", "STRONG_SELL"],
    )
    
    hitl_prompt = HITLPrompt(console)
    human_decision = None
    final_result = None
    
    if compact or no_animation:
        display = CompactAgentDisplay(console)
        
        async for event in run_analysis_streaming(
            ticker,
            llm_model=model,
            guardrail_config=config,
        ):
            event_type = event.get("type", "")
            
            if event_type == "agent_start":
                agent = event.get("agent", "")
                display.print_agent_start(agent)
            
            elif event_type == "agent_thinking":
                if not no_animation:
                    agent = event.get("agent", "")
                    content = event.get("content", "")
                    display.print_agent_thinking(agent, content)
            
            elif event_type == "agent_done":
                agent = event.get("agent", "")
                signal = event.get("signal")
                conf = event.get("confidence", 0)
                if isinstance(conf, float) and conf <= 1:
                    conf = int(conf * 100)
                display.print_agent_done(agent, signal, conf)
            
            elif event_type == "debate_start":
                positions = event.get("positions", {})
                display.print_debate_start(positions)
            
            elif event_type == "consensus_result":
                decision = event.get("decision", "")
                reached = event.get("consensus_reached", False)
                display.print_consensus(decision, reached)
            
            elif event_type == "human_review":
                if not auto_approve:
                    review_data = _build_review_data(event, ticker)
                    human_decision = hitl_prompt.get_review_decision(review_data)
                    
                    if human_decision.get("quit"):
                        return None
                else:
                    print_info(console, "Auto-approving HITL review")
                    human_decision = {
                        "approved": True,
                        "modified_decision": None,
                        "notes": "Auto-approved",
                    }
            
            elif event_type == "final_decision":
                decision = event.get("decision", "HOLD")
                confidence = event.get("confidence", 0)
                requires_review = event.get("requires_review", False)
                display.print_final_decision(decision, confidence, requires_review)
            
            elif event_type == "complete":
                final_result = event.get("result", {})
    
    else:
        display = StreamingAgentDisplay(console, ticker)
        
        with Live(display.create_display(), console=console, refresh_per_second=8) as live:
            async for event in run_analysis_streaming(
                ticker,
                llm_model=model,
                guardrail_config=config,
            ):
                event_type = event.get("type", "")
                
                display.handle_event(event)
                live.update(display.create_display())
                
                if event_type == "human_review":
                    live.stop()
                    
                    if not auto_approve:
                        review_data = _build_review_data(event, ticker)
                        human_decision = hitl_prompt.get_review_decision(review_data)
                        
                        if human_decision.get("quit"):
                            return None
                    else:
                        print_info(console, "Auto-approving HITL review")
                        human_decision = {
                            "approved": True,
                            "modified_decision": None,
                            "notes": "Auto-approved",
                        }
                    
                    live.start()
                
                elif event_type == "complete":
                    final_result = event.get("result", {})
                
                await asyncio.sleep(0.05)
    
    if final_result and human_decision:
        final_result["human_review"] = human_decision
        if human_decision.get("modified_decision"):
            if "manager_decision" not in final_result:
                final_result["manager_decision"] = {}
            final_result["manager_decision"]["decision"] = human_decision["modified_decision"]
            final_result["manager_decision"]["human_modified"] = True
    
    return final_result


def _build_review_data(event: dict, ticker: str) -> dict:
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
        "thesis_summary": state.get("final_thesis", event.get("context", "")),
    }


def _generate_report(result: dict, ticker: str):
    """Generate PDF report from result."""
    try:
        from src.reports.pdf_generator import PDFReportGenerator
        
        print_info(console, "Generating PDF report...")
        
        generator = PDFReportGenerator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_analysis_{timestamp}.pdf"
        
        pdf_path = generator.generate(result, filename)
        print_success(console, f"Report saved: {pdf_path}")
        
    except Exception as e:
        print_error(console, f"Failed to generate report: {e}")


@app.command("chat")
def chat_command(
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model to use"
    ),
):
    """
    Start interactive chat mode with the multi-agent system.
    
    Ask questions about stocks, get insights, or discuss strategies.
    """
    render_logo_static(console)
    
    chat = InteractiveChat(console)
    chat.display_welcome()
    
    try:
        from src.llm.ollama_client import OllamaClient
        client = OllamaClient()
        llm = client.get_llm(model=model, temperature=0.7)
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_prompt = """You are an AI investment analyst assistant for Aegis Flux, a multi-agent hedge fund analysis system.

You can help users with:
- Explaining investment concepts and strategies
- Discussing market trends and analysis techniques
- Answering questions about technical, fundamental, and sentiment analysis
- Providing general financial education

Be concise, professional, and helpful. If asked about specific stock recommendations, 
remind users to run the full analysis using 'aegis analyze <TICKER>' for comprehensive insights.

Do not provide specific buy/sell recommendations in chat - direct users to the analysis command."""

        messages = [SystemMessage(content=system_prompt)]
        
        while True:
            user_input = chat.get_input()
            
            if user_input is None:
                console.print()
                print_info(console, "Exiting chat mode")
                break
            
            messages.append(HumanMessage(content=user_input))
            
            try:
                response = llm.invoke(messages)
                chat.display_response("Aegis AI", response.content, streaming=True)
                messages.append(response)
            except Exception as e:
                print_error(console, f"Error: {e}")
        
    except KeyboardInterrupt:
        console.print()
        print_info(console, "Chat interrupted")
    except Exception as e:
        print_error(console, f"Chat failed: {e}")
        raise typer.Exit(1)


@app.command("status")
def status_command():
    """
    Check system status and connectivity.
    
    Verifies Ollama connection, available models, and database status.
    """
    render_mini_logo(console)
    console.print()
    console.print(create_section_header("SYSTEM STATUS"))
    console.print()
    
    status_table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["muted"],
        expand=True,
    )
    
    status_table.add_column("Component", style="bold")
    status_table.add_column("Status")
    status_table.add_column("Details")
    
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        status_table.add_row(
            "Configuration",
            Text("OK", style=f"bold {COLORS['accent']}"),
            f"Loaded from environment"
        )
    except Exception as e:
        status_table.add_row(
            "Configuration",
            Text("ERROR", style=f"bold {COLORS['danger']}"),
            str(e)[:40]
        )
    
    try:
        from src.llm.ollama_client import OllamaClient
        import httpx
        
        client = OllamaClient()
        response = httpx.get(f"{client.base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "unknown") for m in models[:3]]
            status_table.add_row(
                "Ollama",
                Text("OK", style=f"bold {COLORS['accent']}"),
                f"Models: {', '.join(model_names)}" + ("..." if len(models) > 3 else "")
            )
        else:
            status_table.add_row(
                "Ollama",
                Text("ERROR", style=f"bold {COLORS['danger']}"),
                f"HTTP {response.status_code}"
            )
    except Exception as e:
        status_table.add_row(
            "Ollama",
            Text("OFFLINE", style=f"bold {COLORS['warning']}"),
            str(e)[:40]
        )
    
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        
        if settings.tavily_api_key:
            status_table.add_row(
                "Tavily API",
                Text("OK", style=f"bold {COLORS['accent']}"),
                "API key configured"
            )
        else:
            status_table.add_row(
                "Tavily API",
                Text("NOT SET", style=f"bold {COLORS['warning']}"),
                "Set TAVILY_API_KEY for web search"
            )
    except Exception:
        status_table.add_row(
            "Tavily API",
            Text("UNKNOWN", style=f"bold {COLORS['muted']}"),
            "Could not check"
        )
    
    try:
        from pathlib import Path
        data_dir = Path("./data")
        reports_dir = data_dir / "reports"
        
        if reports_dir.exists():
            report_count = len(list(reports_dir.glob("*.pdf")))
            status_table.add_row(
                "Data Directory",
                Text("OK", style=f"bold {COLORS['accent']}"),
                f"{report_count} reports saved"
            )
        else:
            status_table.add_row(
                "Data Directory",
                Text("EMPTY", style=f"bold {COLORS['warning']}"),
                "No reports yet"
            )
    except Exception:
        status_table.add_row(
            "Data Directory",
            Text("UNKNOWN", style=f"bold {COLORS['muted']}"),
            "Could not check"
        )
    
    console.print(status_table)
    console.print()


@app.command("config")
def config_command(
    show: bool = typer.Option(
        True, "--show/--hide", help="Show current configuration"
    ),
):
    """
    Show or manage configuration settings.
    """
    render_mini_logo(console)
    console.print()
    console.print(create_section_header("CONFIGURATION"))
    console.print()
    
    if show:
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            
            config_table = Table(
                show_header=True,
                header_style=f"bold {COLORS['primary']}",
                border_style=COLORS["muted"],
                expand=True,
            )
            
            config_table.add_column("Setting", style="bold")
            config_table.add_column("Value")
            
            config_table.add_row("Ollama URL", settings.ollama_base_url)
            config_table.add_row("LLM Model", settings.ollama_llm_model)
            config_table.add_row("Embedding Model", settings.ollama_embedding_model)
            config_table.add_row("Agent Temperature", str(settings.agent_temperature))
            config_table.add_row("Max Iterations", str(settings.max_iterations))
            config_table.add_row("Data Directory", str(settings.data_dir))
            config_table.add_row(
                "Tavily API Key",
                "***configured***" if settings.tavily_api_key else "not set"
            )
            
            console.print(config_table)
            console.print()
            
            console.print(
                "Tip: Set configuration via environment variables or .env file",
                style=f"dim {COLORS['muted']}"
            )
            console.print()
            
        except Exception as e:
            print_error(console, f"Failed to load config: {e}")


@app.command("reports")
def reports_command(
    list_all: bool = typer.Option(
        True, "--list/--no-list", "-l", help="List all reports"
    ),
    limit: int = typer.Option(
        10, "--limit", "-n", help="Number of reports to show"
    ),
):
    """
    List and manage analysis reports.
    """
    render_mini_logo(console)
    console.print()
    console.print(create_section_header("REPORTS"))
    console.print()
    
    from pathlib import Path
    
    reports_dir = Path("./data/reports")
    
    if not reports_dir.exists():
        print_warning(console, "No reports directory found")
        return
    
    reports = sorted(reports_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not reports:
        print_info(console, "No reports found")
        return
    
    reports_table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["muted"],
        expand=True,
    )
    
    reports_table.add_column("#", style="dim", width=4)
    reports_table.add_column("Filename", style="bold")
    reports_table.add_column("Size", justify="right")
    reports_table.add_column("Created", justify="right")
    
    for i, report in enumerate(reports[:limit], 1):
        stat = report.stat()
        size_kb = stat.st_size / 1024
        created = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        
        reports_table.add_row(
            str(i),
            report.name,
            f"{size_kb:.1f} KB",
            created,
        )
    
    console.print(reports_table)
    
    if len(reports) > limit:
        console.print(f"\n... and {len(reports) - limit} more reports", style=f"dim {COLORS['muted']}")
    
    console.print()


@app.command("logo")
def logo_command(
    animated: bool = typer.Option(
        True, "--animated/--static", "-a", help="Show animated logo"
    ),
):
    """
    Display the Aegis Flux logo.
    """
    if animated:
        render_logo_animated(console, speed=0.02)
    else:
        render_logo_static(console)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
