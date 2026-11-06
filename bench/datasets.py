import argparse
import random

from bench.types import Request

# Qwen3 ids >= 151643 are special tokens (<|endoftext|>, <|im_start|>, ...)
NUM_REGULAR_TOKENS = 151643


def random_dataset(
    num_requests: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    seed: int,
) -> list[Request]:
    """Random token ids with lengths drawn uniformly from len * [1 - r, 1 + r]."""
    rng = random.Random(seed)

    def sample_len(n: int) -> int:
        return rng.randint(max(1, int(n * (1 - range_ratio))), int(n * (1 + range_ratio)))

    return [
        Request(
            req_id=i,
            prompt_ids=[rng.randrange(NUM_REGULAR_TOKENS) for _ in range(sample_len(input_len))],
            max_tokens=sample_len(output_len),
        )
        for i in range(num_requests)
    ]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-prompts", type=int, default=1000, metavar="N", help="requests to generate"
    )
    parser.add_argument(
        "--input-len", type=int, default=1024, metavar="TOKENS", help="prompt length"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        metavar="TOKENS",
        help="max_tokens per request (fewer if EOS comes first)",
    )
    parser.add_argument(
        "--range-ratio",
        type=float,
        default=0.0,
        metavar="R",
        help="vary lengths uniformly in [len*(1-R), len*(1+R)]; 0 = all equal",
    )


def from_args(args: argparse.Namespace) -> list[Request]:
    return random_dataset(
        args.num_prompts, args.input_len, args.output_len, args.range_ratio, args.seed
    )
