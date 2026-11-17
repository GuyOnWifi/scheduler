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


def shared_prefix_dataset(
    num_requests: int,
    prefix_len: int,
    input_len: int,
    output_len: int,
    range_ratio: float,
    seed: int,
) -> list[Request]:
    """Every prompt is the same prefix_len random tokens (think: one system prompt)
    followed by its own random suffix, drawn like random_dataset's prompts."""
    rng = random.Random(seed)
    prefix = [rng.randrange(NUM_REGULAR_TOKENS) for _ in range(prefix_len)]
    suffixed = random_dataset(num_requests, input_len, output_len, range_ratio, seed + 1)
    return [Request(r.req_id, prefix + r.prompt_ids, r.max_tokens) for r in suffixed]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=["random", "shared-prefix"],
        default="random",
        help="random: random token prompts; shared-prefix: one common --prefix-len "
        "token prefix, then a random suffix of --input-len tokens",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=384,
        metavar="TOKENS",
        help="shared-prefix only: length of the common prefix",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, metavar="N", help="requests to generate"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        metavar="TOKENS",
        help="prompt length (for shared-prefix: the suffix after the prefix)",
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
    if args.dataset == "shared-prefix":
        return shared_prefix_dataset(
            args.num_prompts,
            args.prefix_len,
            args.input_len,
            args.output_len,
            args.range_ratio,
            args.seed,
        )
    return random_dataset(
        args.num_prompts, args.input_len, args.output_len, args.range_ratio, args.seed
    )
