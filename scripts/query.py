#!/usr/bin/env python3
"""Query the inference server."""

import argparse
import json
import logging
import sys

from nanodex.inference.client import QueryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for querying."""
    parser = argparse.ArgumentParser(description="Query nanodex inference server")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="Server endpoint URL",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask (use --interactive for REPL mode)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens in response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive REPL mode",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check server health",
    )

    args = parser.parse_args()

    # Create client
    client = QueryClient(
        endpoint=args.endpoint,
        system_prompt=args.system_prompt,
    )

    # Health check
    if args.health_check:
        if client.health_check():
            print(f" Server is healthy: {args.endpoint}")
            models = client.get_models()
            if models:
                print(f"Available models: {', '.join(models)}")
            return 0
        else:
            print(f" Server is not reachable: {args.endpoint}")
            return 1

    # Interactive mode
    if args.interactive:
        print("=" * 60)
        print("nanodex - Interactive Query Mode")
        print("=" * 60)
        print(f"Endpoint: {args.endpoint}")
        print("Type 'quit' or 'exit' to quit")
        print("=" * 60)
        print("")

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if question.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                result = client.query(
                    question=question,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                if result.get("error"):
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nAnswer: {result['answer']}")
                    if result.get("usage"):
                        usage = result["usage"]
                        print(
                            f"\nTokens: {usage.get('total_tokens', 'N/A')} (prompt: {usage.get('prompt_tokens', 'N/A')}, completion: {usage.get('completion_tokens', 'N/A')})"
                        )

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        return 0

    # Single question mode
    if not args.question:
        parser.error("--question is required (or use --interactive)")

    result = client.query(
        question=args.question,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Print result
    if result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
