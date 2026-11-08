"""Client of jev_server.py (standard library only: no torch on the caller's side).

    client = JEVClient("http://node-2:8200,http://node-2:8201")  # the last line of serve_jev.sh: one URL per GPU
    client.score([(profile, behavior), ...])
    # -> [{"no match": p, "partial match": p, "perfect match": p}, ...], one per pair, in order

profile / behavior: the EVAL_Dataset JSON strings, or a list / dict the server serializes the same way.
score() splits the pairs into chunks of at least MIN_CHUNK (64) pairs, at most one per server, sent at the same time;
a smaller call goes whole to one random server. One rule for every failure of a chunk (connection, HTTP error,
timeout): wait (1, 2, 4, ... up to 60 s) and send it again to the next server, at most RETRIES times (about 4
minutes); only then score() raises.

Smoke test from any node:  python3 jev_client.py http://node-2:8200,http://node-2:8201
"""
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RETRIES = 10
MIN_CHUNK = 64


class JEVClient:
    def __init__(self, urls, timeout: float = 1800):
        """urls: a comma-separated string (serve_jev.sh's output) or a list of server URLs."""
        urls = urls.split(",") if isinstance(urls, str) else urls
        self.urls = [url.strip().rstrip("/") for url in urls if url.strip()]
        if not self.urls:
            raise ValueError("No JEV server URL")
        self.timeout = timeout

    def post(self, url: str, pairs: list) -> list[dict]:
        body = json.dumps({"pairs": [{"profile": profile, "behavior": behavior} for profile, behavior in pairs]},
                          ensure_ascii=False).encode()
        request = urllib.request.Request(url + "/score", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            scores = [result["probability"] for result in json.load(response)["results"]]
        if len(scores) != len(pairs):
            raise RuntimeError(f"{url} returned {len(scores)} scores for {len(pairs)} pairs")
        return scores

    def score_chunk(self, pairs: list, first: int) -> list[dict]:
        """One chunk, first to server `first`, every retry to the next server."""
        for attempt in range(RETRIES):
            url = self.urls[(first + attempt) % len(self.urls)]
            try:
                return self.post(url, pairs)
            except Exception as error:
                if attempt == RETRIES - 1:
                    raise RuntimeError(f"JEV scoring failed {RETRIES} times, last on {url}: {error!r}") from error
                print(f"[jev_client] {url}: {error!r}; retrying", file=sys.stderr, flush=True)
                time.sleep(min(2 ** attempt, 60))

    def score(self, pairs) -> list[dict]:
        pairs = list(pairs)
        if not pairs:
            return []
        # Chunks of at least MIN_CHUNK pairs, at most one per server; a small call (one answer's targets) goes whole
        # to one random server, so the concurrent calls of the reward workers spread over the servers.
        n_chunks = min(len(self.urls), -(-len(pairs) // MIN_CHUNK))
        size = -(-len(pairs) // n_chunks)
        chunks = [pairs[start:start + size] for start in range(0, len(pairs), size)]
        first = random.randrange(len(self.urls))
        if len(chunks) == 1:
            return self.score_chunk(chunks[0], first)
        with ThreadPoolExecutor(len(chunks)) as pool:
            parts = list(pool.map(self.score_chunk, chunks, [first + i for i in range(len(chunks))]))
        return [score for part in parts for score in part]


if __name__ == "__main__":
    profile = [{"Interest": "Energy offers", "Retailer": "ENGIE"}, {"Query": "engie offre déménagement"}]
    behaviors = [{"TimeStamp": "2026-08-28", "Market": "FR", "URL": "particuliers.engie.fr",
                  "PageTitle": "Découvrez les offres d'énergie pour votre déménagement | ENGIE"},
                 {"TimeStamp": "2026-08-12", "Market": "FR", "URL": "outlook.office.com",
                  "PageTitle": "Microsoft Authentication"}]
    for behavior, probability in zip(behaviors, JEVClient(sys.argv[1]).score([(profile, b) for b in behaviors])):
        print(behavior["PageTitle"], "->", probability)
