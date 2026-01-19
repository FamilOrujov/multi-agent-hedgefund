#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from pprint import pprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


AGENT_NAMES = [
    "data_scout",
    "technical_analyst",
    "fundamental_analyst",
    "sentiment_analyst",
    "portfolio_manager",
]


def debug_agent(agent_name: str, ticker: str, verbose: bool = False):
    """Debug a specific agent."""
    from src.llm.ollama_client import OllamaClient
    from src.api.dependencies import get_agent_by_name
    
    print(f"\n{'='*60}")
    print(f"  Debugging: {agent_name}")
    print(f"  Ticker: {ticker}")
    print(f"{'='*60}\n")
    
    # Initialize client
    print("🔌 Connecting to Ollama...")
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama is not available!")
        return
    
    print(f"✅ Connected to Ollama (model: {client.model})")
    
    # Get agent
    agent = get_agent_by_name(agent_name, client)
    if not agent:
        print(f"❌ Agent '{agent_name}' not found!")
        print(f"   Available agents: {', '.join(AGENT_NAMES)}")
        return
    
    print(f"\n📋 Agent Info:")
    print(f"   Name: {agent.name}")
    print(f"   Role: {agent.role}")
    print(f"   Model: {agent.model or 'default'}")
    print(f"   Temperature: {agent.temperature}")
    
    if verbose:
        print(f"\n📝 System Prompt:")
        print("-" * 40)
        print(agent.system_prompt)
        print("-" * 40)
    
    # Build initial state
    state = {"ticker": ticker}
    
    # For non-data-scout agents, first run data scout
    if agent_name != "data_scout":
        print(f"\n🔍 Running Data Scout first to populate state...")
        data_scout = get_agent_by_name("data_scout", client)
        start_time = time.time()
        state = data_scout.analyze(state)
        scout_time = time.time() - start_time
        print(f"   ✅ Data Scout complete ({scout_time:.2f}s)")
        
        if verbose:
            print("\n📦 State after Data Scout:")
            print(f"   Keys: {list(state.keys())}")
    
    # Run the target agent
    print(f"\n🤖 Running {agent_name}...")
    start_time = time.time()
    
    try:
        result = agent.analyze(state)
        elapsed = time.time() - start_time
        print(f"   ✅ Complete ({elapsed:.2f}s)")
        
        # Extract relevant output
        print(f"\n📊 Results:")
        
        if agent_name == "data_scout":
            print(f"\n   Summary:")
            print(f"   {result.get('data_scout_summary', 'N/A')[:500]}...")
            print(f"\n   Price Summary: {result.get('price_summary', {})}")
            print(f"\n   News Headlines: {result.get('news_data', [])[:5]}")
            
        elif agent_name == "technical_analyst":
            signal = result.get("technical_signal", {})
            print(f"\n   Signal: {signal.get('signal', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 'N/A')}")
            print(f"   Bullish Count: {signal.get('bullish_count', 0)}")
            print(f"   Bearish Count: {signal.get('bearish_count', 0)}")
            print(f"\n   Analysis:")
            print(f"   {result.get('technical_analysis', 'N/A')[:800]}...")
            
        elif agent_name == "fundamental_analyst":
            signal = result.get("fundamental_signal", {})
            print(f"\n   Signal: {signal.get('signal', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 'N/A')}")
            health = result.get("health_assessment", {})
            print(f"   Health: {health.get('health', 'N/A')} (Score: {health.get('score', 'N/A')})")
            print(f"\n   Analysis:")
            print(f"   {result.get('fundamental_analysis', 'N/A')[:800]}...")
            
        elif agent_name == "sentiment_analyst":
            signal = result.get("sentiment_signal", {})
            print(f"\n   Signal: {signal.get('signal', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 'N/A')}")
            print(f"   Sentiment Score: {signal.get('score', 'N/A')}")
            print(f"\n   Analysis:")
            print(f"   {result.get('sentiment_analysis', 'N/A')[:800]}...")
            
        elif agent_name == "portfolio_manager":
            decision = result.get("manager_decision", {})
            print(f"\n   Decision: {decision.get('decision', 'N/A')}")
            print(f"   Confidence: {decision.get('confidence', 'N/A')}%")
            print(f"   Position Size: {decision.get('position_size', 'N/A')}")
            print(f"   Consensus: {decision.get('has_consensus', 'N/A')}")
            print(f"\n   Thesis:")
            print(f"   {result.get('final_thesis', 'N/A')[:1000]}...")
        
        if verbose:
            print(f"\n📦 Full Result Keys: {list(result.keys())}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Debug individual agents")
    parser.add_argument("agent", choices=AGENT_NAMES, help="Agent to debug")
    parser.add_argument("ticker", help="Stock ticker to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    debug_agent(args.agent, args.ticker.upper(), args.verbose)


if __name__ == "__main__":
    main()
