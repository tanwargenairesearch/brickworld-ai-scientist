"""Offline tests of the proposer's retry rules (no model is called).
Infrastructure errors are retried without using up the model's 3 attempts; invalid replies use one."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import proposer  # noqa: E402
from brickworld.space import BASELINE  # noqa: E402

proposer.time.sleep = lambda s: None
GOOD = {"analysis": "ok", "proposals": [{"hypothesis": f"h{i}", "config": dict(BASELINE)} for i in range(4)]}


def run(replies):
    seq = iter(replies)

    def fake(model, system, user):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r, json.dumps(r), {"output_tokens": 1}

    proposer._call_cli = fake
    with tempfile.TemporaryDirectory() as d:
        try:
            props = proposer.propose("cli", "m", [], 1, 8, 4, "", Path(d))
            rec = json.loads((Path(d) / "round_01.json").read_text())
            return props, rec
        except RuntimeError as e:
            return None, str(e)


def test_infra_errors_do_not_use_attempts():
    props, rec = run([proposer.InfraError("rate limit")] * 5 + [GOOD])
    assert props and rec["attempt"] == 1 and len(rec["infra_errors"]) == 5


def test_invalid_reply_uses_an_attempt_and_feeds_back():
    bad = {"analysis": "x", "proposals": []}
    props, rec = run([bad, proposer.ModelError("refusal"), GOOD])
    assert props and rec["attempt"] == 3 and len(rec["rejected_attempts"]) == 2
    assert "Your previous reply was rejected" in rec["prompt"]


def test_three_invalid_replies_end_the_round():
    bad = {"analysis": "x", "proposals": [{"hypothesis": "h", "config": {"lr": 1}}] * 4}
    props, msg = run([bad, bad, bad, GOOD])
    assert props is None and "failed 3 times" in msg


def test_cli_error_classification():
    assert proposer._INFRA_WORDS.search("cli error (error): Claude AI usage limit reached")
    assert proposer._INFRA_WORDS.search("API Error: 529 overloaded")
    assert not proposer._INFRA_WORDS.search("cli error (error_max_structured_output_retries): bad output")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
