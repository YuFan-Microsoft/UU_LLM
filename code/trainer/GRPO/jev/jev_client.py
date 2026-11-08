"""Client of jev_server.py (standard library only: no torch on the caller's side).

    client = JEVClient("http://node-2:8200")
    client.score([(profile, behavior), ...])
    # -> [{"no match": p, "partial match": p, "perfect match": p}, ...], one per pair, in order

profile / behavior: the EVAL_Dataset JSON strings, or a list / dict the server serializes the same way.
A connection failure is retried for about 3 minutes (backoff up to 60 s), the time a restarted server takes to come
back; an HTTP error (bad request, failed GPU round) or a read timeout (`timeout`, 1 h) is raised at once.

Smoke test from any node:  python3 jev_client.py http://node-2:8200
"""
import json
import sys
import time
import urllib.error
import urllib.request


class JEVClient:
    def __init__(self, url: str, timeout: float = 3600, retries: int = 9):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def score(self, pairs) -> list[dict]:
        body = json.dumps({"pairs": [{"profile": profile, "behavior": behavior} for profile, behavior in pairs]},
                          ensure_ascii=False).encode()
        request = urllib.request.Request(self.url + "/score", data=body, headers={"Content-Type": "application/json"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return [result["probability"] for result in json.load(response)["results"]]
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"JEV server {error.code}: {error.read().decode(errors='replace')}") from None
            # A read timeout is not retried: the server is still scoring, and a resend would score everything twice.
            except (urllib.error.URLError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 60))


if __name__ == "__main__":
    profile = [{"Interest": "Energy offers", "Retailer": "ENGIE"}, {"Query": "engie offre déménagement"}]
    behaviors = [{"TimeStamp": "2026-08-28", "Market": "FR", "URL": "particuliers.engie.fr",
                  "PageTitle": "Découvrez les offres d'énergie pour votre déménagement | ENGIE"},
                 {"TimeStamp": "2026-08-12", "Market": "FR", "URL": "outlook.office.com",
                  "PageTitle": "Microsoft Authentication"}]
    for behavior, probability in zip(behaviors, JEVClient(sys.argv[1]).score([(profile, b) for b in behaviors])):
        print(behavior["PageTitle"], "->", probability)
