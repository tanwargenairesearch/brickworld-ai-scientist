"""Check that the Anthropic API key in the environment works and has credit.

Lists models (free), then sends one tiny message to each proposer model.
Never prints the key. Cost: well under one cent in total.
"""
import sys

import anthropic

MODELS = ["claude-sonnet-5-5", "claude-opus-5-5"]
PRICE = {"claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}  # $/MTok in, out

client = anthropic.Anthropic()

try:
    ids = [m.id for m in client.models.list()]
except anthropic.AuthenticationError:
    sys.exit("Authentication failed: the API key is invalid or revoked.")
except anthropic.APIConnectionError as e:
    sys.exit(f"Could not reach the API: {e}")
print(f"Key accepted; {len(ids)} models visible.")
for m in MODELS:
    print(f"  {m}: {'available' if m in ids else 'NOT listed'}")

total = 0.0
for m in MODELS:
    try:
        r = client.messages.create(
            model=m,
            max_tokens=1024,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
        )
    except anthropic.BadRequestError as e:
        print(f"  {m}: request rejected (400): {e.message}")
        continue
    except anthropic.PermissionDeniedError as e:
        print(f"  {m}: permission denied: {e.message}")
        continue
    except anthropic.RateLimitError as e:
        print(f"  {m}: rate limited: {e.message}")
        continue
    except anthropic.APIStatusError as e:
        print(f"  {m}: API error {e.status_code}: {e.message}")
        continue
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    pin, pout = PRICE[m]
    cost = r.usage.input_tokens * pin / 1e6 + r.usage.output_tokens * pout / 1e6
    total += cost
    print(f"  {m}: stop={r.stop_reason} reply={text!r} in={r.usage.input_tokens} "
          f"out={r.usage.output_tokens} cost=${cost:.5f}")
print(f"Total cost of this check: ${total:.5f}")
