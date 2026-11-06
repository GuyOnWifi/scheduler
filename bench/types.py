from dataclasses import dataclass, field


@dataclass(frozen=True)
class Request:
    req_id: int
    prompt_ids: list[int]
    max_tokens: int  # forced output length (benchmarks ignore EOS)
    arrival_time: float = 0.0  # seconds after benchmark start


@dataclass(frozen=True)
class TokenEvent:
    req_id: int
    token_id: int
    finished: bool


@dataclass
class RequestRecord:
    """Timestamps for one request, in seconds since benchmark start."""

    req_id: int
    prompt_len: int
    arrival_time: float
    token_times: list[float] = field(default_factory=list)

    @property
    def output_len(self) -> int:
        return len(self.token_times)

    @property
    def ttft(self) -> float:
        return self.token_times[0] - self.arrival_time

    @property
    def e2el(self) -> float:
        return self.token_times[-1] - self.arrival_time

    @property
    def tpot(self) -> float | None:
        # average decode time per token after the first; undefined for 1-token outputs
        if self.output_len < 2:
            return None
        return (self.e2el - self.ttft) / (self.output_len - 1)

    @property
    def itls(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]
