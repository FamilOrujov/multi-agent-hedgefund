#!/usr/bin/env python3

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def benchmark_agent(agent_name: str, state: dict, client, runs: int = 1) -> dict:
    """Benchmark a single agent."""
    from src.api.dependencies import get_agent_by_name
    
    agent = get_agent_by_name(agent_name, client)
    times = []
    
    for i in range(runs):
        start = time.time()
        result = agent.analyze(state.copy())
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        "agent": agent_name,
        "runs": runs,
        "times": times,
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if runs > 1 else 0,
    }


def benchmark_full_pipeline(ticker: str, runs: int = 1) -> dict:
    """Benchmark the full analysis pipeline."""
    from src.llm.ollama_client import OllamaClient
    from src.graph.workflow import run_analysis
    from src.graph.guardrails import GuardrailConfig
    
    print(f"\n{'='*60}")
    print(f"  AEGIS FLUX BENCHMARK")
    print(f"  Ticker: {ticker} | Runs: {runs}")
    print(f"{'='*60}\n")
    
    # Initialize
    print("🔌 Initializing Ollama client...")
    client = OllamaClient()
    
    if not client.is_available():
        print("❌ Ollama is not available!")
        return {}
    
    print(f"✅ Connected (model: {client.model})")
    
    results = {
        "ticker": ticker,
        "runs": runs,
        "agents": {},
        "pipeline": {},
    }
    
    # Benchmark individual agents
    print("\n📊 Benchmarking Individual Agents...")
    print("-" * 50)
    
    # First, run data scout to get base state
    from src.api.dependencies import get_agent_by_name
    
    print(f"  Data Scout... ", end="", flush=True)
    state = {"ticker": ticker}
    scout_start = time.time()
    data_scout = get_agent_by_name("data_scout", client)
    state = data_scout.analyze(state)
    scout_time = time.time() - scout_start
    print(f"{scout_time:.2f}s")
    results["agents"]["data_scout"] = {"mean": scout_time, "min": scout_time, "max": scout_time}
    
    # Benchmark other agents
    for agent_name in ["technical_analyst", "fundamental_analyst", "sentiment_analyst", "portfolio_manager"]:
        print(f"  {agent_name}... ", end="", flush=True)
        bench = benchmark_agent(agent_name, state, client, runs)
        print(f"{bench['mean']:.2f}s (±{bench['stdev']:.2f}s)")
        results["agents"][agent_name] = bench
    
    # Benchmark full pipeline
    print("\n📊 Benchmarking Full Pipeline...")
    print("-" * 50)
    
    pipeline_times = []
    guardrail_config = GuardrailConfig(
        enable_confidence_check=True,
        enable_consistency_check=True,
    )
    
    for i in range(runs):
        print(f"  Run {i+1}/{runs}... ", end="", flush=True)
        start = time.time()
        result = run_analysis(
            ticker=ticker,
            ollama_client=client,
            analysis_depth="standard",
            guardrail_config=guardrail_config,
            enable_hitl=False,
        )
        elapsed = time.time() - start
        pipeline_times.append(elapsed)
        print(f"{elapsed:.2f}s")
    
    results["pipeline"] = {
        "runs": runs,
        "times": pipeline_times,
        "mean": statistics.mean(pipeline_times),
        "min": min(pipeline_times),
        "max": max(pipeline_times),
        "stdev": statistics.stdev(pipeline_times) if runs > 1 else 0,
    }
    
    # Summary
    print(f"\n{'='*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    print("Agent Times (mean):")
    total_agent_time = 0
    for agent_name, data in results["agents"].items():
        mean_time = data.get("mean", data.get("min", 0))
        total_agent_time += mean_time
        print(f"  {agent_name:25} {mean_time:6.2f}s")
    
    print(f"\n  {'Total (sequential)':25} {total_agent_time:6.2f}s")
    print(f"\nFull Pipeline:")
    print(f"  Mean: {results['pipeline']['mean']:.2f}s")
    print(f"  Min:  {results['pipeline']['min']:.2f}s")
    print(f"  Max:  {results['pipeline']['max']:.2f}s")
    if runs > 1:
        print(f"  StdDev: {results['pipeline']['stdev']:.2f}s")
    
    # Save results
    output_file = project_root / "data" / f"benchmark_{ticker}_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Aegis Flux analysis pipeline")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker to benchmark")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    args = parser.parse_args()

    benchmark_full_pipeline(args.ticker.upper(), args.runs)


if __name__ == "__main__":
    main()
