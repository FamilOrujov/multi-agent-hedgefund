#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_health(client: httpx.Client) -> bool:
    """Test health endpoints."""
    print_section("Health Check")
    
    # Basic health
    resp = client.get("/api/v1/health")
    print(f"GET /api/v1/health: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    
    if resp.status_code != 200:
        print("❌ Health check failed!")
        return False
    print("✅ Basic health OK")
    
    # Ollama health
    resp = client.get("/api/v1/health/ollama")
    print(f"\nGET /api/v1/health/ollama: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    
    # Tavily health
    resp = client.get("/api/v1/health/tavily")
    print(f"\nGET /api/v1/health/tavily: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    
    return True


def test_agents(client: httpx.Client) -> bool:
    """Test agents endpoints."""
    print_section("Agents Endpoints")
    
    # List agents
    resp = client.get("/api/v1/agents")
    print(f"GET /api/v1/agents: {resp.status_code}")
    data = resp.json()
    print(f"  Found {data['count']} agents:")
    for agent in data['agents']:
        print(f"    - {agent['name']}: {agent['role']}")
    
    if resp.status_code != 200:
        print("❌ List agents failed!")
        return False
    print("✅ List agents OK")
    
    # Get specific agent
    resp = client.get("/api/v1/agents/technical_analyst")
    print(f"\nGET /api/v1/agents/technical_analyst: {resp.status_code}")
    print(f"  Response: {resp.json()}")
    
    # Get agent config
    resp = client.get("/api/v1/agents/technical_analyst/config")
    print(f"\nGET /api/v1/agents/technical_analyst/config: {resp.status_code}")
    config = resp.json()
    print(f"  System prompt preview: {config.get('system_prompt', '')[:100]}...")
    
    return True


def test_analysis(client: httpx.Client, ticker: str) -> bool:
    """Test analysis endpoint."""
    print_section(f"Analysis Test ({ticker})")
    
    print(f"Starting analysis for {ticker}...")
    print("(This may take a few minutes)\n")
    
    start_time = time.time()
    
    try:
        resp = client.post(
            "/api/v1/analyze",
            json={"ticker": ticker, "depth": "quick", "enable_hitl": False},
            timeout=300.0,  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        print(f"POST /api/v1/analyze: {resp.status_code}")
        print(f"  Time elapsed: {elapsed:.2f}s")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n  Analysis ID: {data['id']}")
            print(f"  Status: {data['status']}")
            print(f"  Decision: {data.get('decision', 'N/A')}")
            print(f"  Confidence: {data.get('confidence', 'N/A')}%")
            print(f"  Position: {data.get('position_size', 'N/A')}")
            print(f"\n  Signals:")
            print(f"    Technical: {data.get('technical_signal', {}).get('signal', 'N/A')}")
            print(f"    Fundamental: {data.get('fundamental_signal', {}).get('signal', 'N/A')}")
            print(f"    Sentiment: {data.get('sentiment_signal', {}).get('signal', 'N/A')}")
            print("\n✅ Analysis completed successfully!")
            return True
        else:
            print(f"❌ Analysis failed: {resp.text}")
            return False
            
    except httpx.TimeoutException:
        print("❌ Analysis timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Aegis Flux API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to test analysis with")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip the analysis test (slow)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   AEGIS FLUX API TESTER                      ║
╠══════════════════════════════════════════════════════════════╣
║  Base URL: {args.base_url:<47} ║
║  Ticker:   {args.ticker:<47} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        # Test health
        if not test_health(client):
            print("\n❌ Health check failed - is the server running?")
            sys.exit(1)
        
        # Test agents
        if not test_agents(client):
            print("\n❌ Agents test failed")
            sys.exit(1)
        
        # Test analysis (optional)
        if not args.skip_analysis:
            if not test_analysis(client, args.ticker):
                print("\n❌ Analysis test failed")
                sys.exit(1)
        else:
            print("\n⏭️  Skipping analysis test (--skip-analysis)")
    
    print_section("All Tests Passed! ✅")


if __name__ == "__main__":
    main()
