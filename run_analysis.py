#!/usr/bin/env python3
"""
Aegis Flux - Multi-Agent Investment Analysis System

Interactive CLI - Claude Code Style

Usage:
    python run_analysis.py      # Launch interactive CLI
    
Commands (inside CLI):
    /analyze TSLA              # Analyze a stock
    /doctor                    # Run system diagnostics
    /agents                    # View agent info
    /help                      # Show all commands
    /quit                      # Exit
    
Or just type a message to chat with the AI agents.
"""

from src.cli.interactive import main

if __name__ == "__main__":
    main()
