"""Simple CLI for error_map"""

import asyncio
import argparse
from . import ErrorMap


async def main():
    parser = argparse.ArgumentParser(description="Async Error Analysis with Model List Support")
    parser.add_argument("--models", nargs="+", help="Models to analyze")
    parser.add_argument("--ratio", type=float, default=0.1, help="Error sampling ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inference-type", choices=["litellm", "litellm-mock"], 
                       default="litellm-mock", help="Inference type")
    parser.add_argument("--exp-id", help="Experiment ID")
    parser.add_argument("--max-workers", type=int, default=100, 
                       help="Max concurrent inference workers (default: 100)")
    parser.add_argument("--datasets", nargs="+", help="Dataset names to process")
    parser.add_argument("--data-path", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", help="Path to outputs")
    parser.add_argument("--judge", help="Judge model")
    parser.add_argument("--provider", help="Inference provider", choices=["azure", "rits"])
    parser.add_argument("--no-use-correct-predictions", action="store_false", dest="use_correct_predictions", help="Disable adding correct predictions from other models (enabled by default)")
    args = parser.parse_args()
    
    error_map = ErrorMap(
        inference_type=args.inference_type,
        exp_id=args.exp_id,
        max_workers=args.max_workers,
        data_path=args.data_path,
        output_dir=args.output_dir,
        datasets=args.datasets,
        judge=args.judge,
        provider=args.provider,
        seed=args.seed,
        use_correct_predictions=args.use_correct_predictions,
        models=args.models,
        ratio=args.ratio,
    )
    
    results = await error_map.run()
    
    print(f"\n✅ Complete! Experiment: {results['exp_id']}")
    print(f"Records: {results['total_records']}, Errors: {results['error_records']}")


def cli_main():
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()