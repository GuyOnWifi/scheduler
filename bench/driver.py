from itertools import accumulate
from collections import deque
import random
import time

from bench.engine import Engine
from bench.types import Request, RequestRecord


def arrival_times(
    n: int, rate: float, burstiness: float, rng: random.Random
) -> list[float]:
    """Arrival time (seconds after start) of each of n requests.

    Average rate is `rate` req/s. Gaps between arrivals are gamma-distributed with
    shape=burstiness and scale=1/(rate*burstiness): burstiness=1 is a Poisson process,
    <1 is burstier, >1 more uniform. rate=inf means every request arrives at 0.

    Hint: rng.gammavariate(alpha, beta) is shape/scale.
    """
    if rate == float("inf"):
        return [0.0] * n

    scale = 1 / (
        burstiness * rate
    )  # so that alpha * beta is the mean, which should be 1/rate
    dist = [rng.gammavariate(burstiness, scale) for _ in range(n)]
    return list(
        accumulate(
            dist,
        )
    )


def drive(engine: Engine, requests: list[Request]) -> tuple[list[RequestRecord], float]:
    """Feed requests to the engine at their arrival_time and step until all finish.

    Returns one RequestRecord per request (with a timestamp per generated token) and
    the total wall-clock duration in seconds. All timestamps are relative to the same
    start time, taken with time.perf_counter().

    Think about: what to do when nothing has arrived yet and the engine is idle,
    and when exactly each token's timestamp should be taken.
    """

    pending = deque(sorted(requests, key=lambda x: x.arrival_time))
    start = time.perf_counter()

    records = {
        r.req_id: RequestRecord(r.req_id, len(r.prompt_ids), r.arrival_time, [])
        for r in requests
    }

    while True:
        cur = time.perf_counter()
        delta = cur - start

        while pending and pending[0].arrival_time <= delta:
            req = pending.popleft()
            engine.add_request(req)

        if engine.has_unfinished():
            evts = engine.step()
            ts = time.perf_counter() - start
            for e in evts:
                records[e.req_id].token_times.append(ts)

        elif pending:
            time.sleep(
                max(0.0, pending[0].arrival_time - (time.perf_counter() - start))
            )
        else:
            return (list(records.values()), time.perf_counter() - start)
