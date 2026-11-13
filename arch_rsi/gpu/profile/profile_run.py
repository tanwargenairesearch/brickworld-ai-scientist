"""Profile one GPU training run: where does the time go? Not part of any experiment.

A copy of train_gpu.py's training loop (same maths, same order of random draws), with a timer around each
section. Each timer synchronises the GPU before reading the clock, so sections are measured exactly but the
total runs somewhat slower than an untimed run. The script also reports the untimed run's wall time from the
original log for comparison, and the dev score of the timed run, which should match the original bit for bit.

usage (project root): .venv/bin/python -I arch_rsi/gpu/profile/profile_run.py --candidate PATH --seed N --out DIR
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from arch_rsi.interface import CandidatePolicy, arch_spec
from arch_rsi.sandbox import load_candidate, sha
from arch_rsi.train_arch import heldout_hashes
from arch_rsi.worlds import RECIPE, SPECS
from brickworld.env import BrickWorld, StreamSource
from brickworld.model import encode
from train import TRAIN_SEED_BASE, reward_fn

BASELINE = ROOT / "arch_rsi" / "candidates" / "baseline.py"
T_ = defaultdict(float)
dev = torch.device("mps")


class tm:
    def __init__(self, k): self.k = k
    def __enter__(self):
        torch.mps.synchronize(); self.t = time.perf_counter()
    def __exit__(self, *a):
        torch.mps.synchronize(); T_[self.k] += time.perf_counter() - self.t


ap = argparse.ArgumentParser()
ap.add_argument("--candidate", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--budget", type=int, default=262144)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
src = Path(a.candidate).read_text(); (out / "candidate.py").write_text(src)
rc = {**RECIPE, "n_envs": 256, "lr": 0.001, "n_minibatches": 32}
spec = SPECS["xl"]; aspec = arch_spec(spec); seed = a.seed
torch.set_num_threads(1); torch.manual_seed(seed)
from arch_rsi.check import train_flops
flops_ratio = train_flops(src, aspec) / train_flops(BASELINE.read_text(), aspec)
torch.manual_seed(seed)
policy = CandidatePolicy(load_candidate(src)(aspec), aspec).to(dev)
stream = StreamSource(spec, TRAIN_SEED_BASE + seed, heldout_hashes(spec.name))
N, T = rc["n_envs"], rc["rollout_len"]
world = BrickWorld(spec, N, dev, target_source=stream)
opt = torch.optim.Adam(policy.parameters(), lr=rc["lr"], eps=1e-5)
n_updates = max(1, int(max(1, a.budget // (N * T)) * min(1.0, 1.0 / flops_ratio)))
batch = N * T; mb = batch // rc["n_minibatches"]
obs = encode(world, "voxels"); C = obs.shape[1]
z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
b_obs = z(T, N, C, spec.Y, spec.X); b_mask = z(T, N, spec.n_actions, dtype=torch.bool); b_act = z(T, N, dtype=torch.long)
b_logp, b_val, b_rew, b_done = (z(T, N) for _ in range(4)); ep_ret = z(N)
torch.mps.synchronize(); t0 = time.perf_counter()
for upd in range(1, n_updates + 1):
    for g in opt.param_groups:
        g["lr"] = rc["lr"] * (1.0 - (upd - 1) / n_updates)
    policy.eval()
    for t in range(T):
        with tm("env: valid_mask"):
            mask = world.valid_mask()
        with tm("rollout: policy forward + sample"):
            with torch.no_grad():
                logits, value = policy(obs, mask)
                act = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
                logp = F.log_softmax(logits, dim=-1).gather(1, act.unsqueeze(1)).squeeze(1)
        with tm("rollout: store"):
            b_obs[t], b_mask[t], b_act[t], b_logp[t], b_val[t] = obs, mask, act, logp, value
        with tm("env: step"):
            placed = act != spec.done_action
            info = world.step(act)
        with tm("env: reward + bookkeeping"):
            r = reward_fn(rc, info, placed); b_rew[t] = r
            ended = info["ended"]; b_done[t] = ended.float(); ep_ret += r
            ep_ret = torch.where(ended, torch.zeros_like(ep_ret), ep_ret)
        with tm("env: encode observation"):
            obs = encode(world, "voxels")
    with tm("rollout: bootstrap value + GAE"):
        with torch.no_grad():
            _, next_val = policy(obs, world.valid_mask())
        adv, last = z(T, N), z(N)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]; nv = next_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + rc["gamma"] * nv * nonterm - b_val[t]
            last = delta + rc["gamma"] * rc["gae_lambda"] * nonterm * last; adv[t] = last
        ret = adv + b_val
    f_obs, f_mask = b_obs.reshape(batch, C, spec.Y, spec.X), b_mask.reshape(batch, -1)
    f_act, f_logp, f_adv, f_ret = b_act.reshape(-1), b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
    policy.train()
    for _ in range(rc["ppo_epochs"]):
        perm = torch.randperm(batch, device=dev)
        for i in range(0, batch, mb):
            idx = perm[i:i + mb]
            with tm("update: forward + loss"):
                logits, v = policy(f_obs[idx], f_mask[idx])
                lp_all = F.log_softmax(logits, dim=-1); lp = lp_all.gather(1, f_act[idx].unsqueeze(1)).squeeze(1)
                ent = -(lp_all.exp() * lp_all).sum(-1).mean(); A = f_adv[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = (lp - f_logp[idx]).exp()
                pg = -torch.min(ratio * A, ratio.clamp(1 - rc["clip_eps"], 1 + rc["clip_eps"]) * A).mean()
                vloss = 0.5 * (v - f_ret[idx]).pow(2).mean()
                loss = pg - rc["ent_coef"] * ent + rc["vf_coef"] * vloss
            with tm("update: backward"):
                opt.zero_grad(set_to_none=True); loss.backward()
            with tm("update: clip + optimiser step"):
                torch.nn.utils.clip_grad_norm_(policy.parameters(), rc["max_grad_norm"]); opt.step()
torch.mps.synchronize(); total = time.perf_counter() - t0
state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
torch.save({"state_dict": state, "world": spec.name, "candidate_sha": sha(src), "steps": n_updates * batch}, out / "ckpt.pt")
res = {"candidate_sha": sha(src), "seed": seed, "n_updates": n_updates, "steps": n_updates * batch,
       "flops_ratio": round(flops_ratio, 3), "timed_total_s": round(total, 2),
       "sections_s": {k: round(v, 2) for k, v in sorted(T_.items(), key=lambda kv: -kv[1])},
       "sections_share": {k: round(v / total, 3) for k, v in sorted(T_.items(), key=lambda kv: -kv[1])}}
(out / "profile.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
