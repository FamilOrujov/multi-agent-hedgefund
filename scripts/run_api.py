#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Aegis Flux API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    AEGIS FLUX API SERVER                     ║
╠══════════════════════════════════════════════════════════════╣
║  Starting server...                                          ║
║  Host: {args.host:<52} ║
║  Port: {args.port:<52} ║
║  Reload: {str(args.reload):<50} ║
║                                                              ║
║  Documentation: http://{args.host}:{args.port}/docs{' ' * (35 - len(str(args.port)))}║
║  Health Check:  http://{args.host}:{args.port}/api/v1/health{' ' * (22 - len(str(args.port)))}║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
