"""VISUALIZATION ONLY, never used for any score, selection or campaign analysis.

A line-for-line copy of the frozen training loop in arch_rsi/train_arch.py (same RECIPE, reward, design stream,
held-out-design rejection, seed and compute-matched update count). The only addition: it keeps copies of the
policy weights at 11 evenly spaced points in training (update 0 = untrained, then every 10% of the updates),
so a video can show what the network builds as it learns. Taking a copy of the weights draws no random numbers,
so training is unchanged; the final snapshot is checked against the frozen run's own dev score in render_learning.py.

usage (project root): .venv/bin/python -I arch_rsi/viz/train_snapshots.py --candidate PATH --world xl --seed 0 \
                          --budget 131072 --out arch_rsi/viz/runs/NAME
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.train_arch import heldout_hashes  # noqa: E402
from arch_rsi.worlds import RECIPE, SPECS  # noqa: E402
from brickworld.env import BrickWorld, StreamSource  # noqa: E402
from brickworld.model import encode  # noqa: E402
from train import TRAIN_SEED_BASE, reward_fn  # noqa: E402

HERE = Path(__file__).resolve().parent
N_SNAP = 10  # snapshots at k/N_SNAP of the updates, k = 0..N_SNAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.candidate).read_text()
    (out / "candidate.py").write_text(src)
    rc = dict(RECIPE)
    spec = SPECS[args.world]
    aspec = arch_spec(spec)
    seed = args.seed
    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    t0 = time.time()
    from arch_rsi.check import train_flops
    flops_ratio = train_flops(src, aspec) / train_flops((ROOT / "arch_rsi" / "candidates" / "baseline.py").read_text(), aspec)
    torch.manual_seed(seed)
    build = load_candidate(src)
    policy = CandidatePolicy(build(aspec), aspec).to(device)

    stream = StreamSource(spec, TRAIN_SEED_BASE + seed, heldout_hashes(spec.name))
    N, T = rc["n_envs"], rc["rollout_len"]
    world = BrickWorld(spec, N, device, target_source=stream)
    opt = torch.optim.Adam(policy.parameters(), lr=rc["lr"], eps=1e-5)
    base_updates = max(1, int(args.budget) // (N * T))
    n_updates = max(1, int(base_updates * min(1.0, 1.0 / flops_ratio)))
    batch = N * T
    mb = batch // rc["n_minibatches"]
    snap_at = sorted({round(k * n_updates / N_SNAP) for k in range(N_SNAP + 1)})
    snaps = {}

    def snapshot(u):  # visualization only: a copy of the weights, no random numbers drawn
        snaps[u] = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    snapshot(0)
    obs = encode(world, "voxels")
    C = obs.shape[1]
    b_obs = torch.zeros(T, N, C, spec.Y, spec.X)
    b_mask = torch.zeros(T, N, spec.n_actions, dtype=torch.bool)
    b_act = torch.zeros(T, N, dtype=torch.long)
    b_logp, b_val, b_rew, b_done = (torch.zeros(T, N) for _ in range(4))
    ep_ret = torch.zeros(N)
    curve, status = [], "ok"
    for upd in range(1, n_updates + 1):
        if rc["lr_schedule"] == "linear_decay":
            for g in opt.param_groups:
                g["lr"] = rc["lr"] * (1.0 - (upd - 1) / n_updates)
        n_end = n_exact = 0
        len_sum = ret_sum = 0.0
        policy.eval()
        for t in range(T):
            mask = world.valid_mask()
            with torch.no_grad():
                logits, value = policy(obs, mask)
                a = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
                logp = F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(1)).squeeze(1)
            b_obs[t], b_mask[t], b_act[t], b_logp[t], b_val[t] = obs, mask, a, logp, value
            placed = a != spec.done_action
            info = world.step(a)
            r = reward_fn(rc, info, placed)
            b_rew[t] = r
            ended = info["ended"]
            b_done[t] = ended.float()
            ep_ret += r
            if ended.any():
                n_end += int(ended.sum())
                n_exact += int(info["exact"].sum())
                len_sum += float(info["ep_len"][ended].float().sum())
                ret_sum += float(ep_ret[ended].sum())
                ep_ret[ended] = 0.0
            obs = encode(world, "voxels")
        with torch.no_grad():
            _, next_val = policy(obs, world.valid_mask())
        adv = torch.zeros(T, N)
        last = torch.zeros(N)
        gamma, lam = rc["gamma"], rc["gae_lambda"]
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            nv = next_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + gamma * nv * nonterm - b_val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + b_val
        f_obs, f_mask = b_obs.reshape(batch, C, spec.Y, spec.X), b_mask.reshape(batch, -1)
        f_act, f_logp, f_adv, f_ret = b_act.reshape(-1), b_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        policy.train()
        for _ in range(rc["ppo_epochs"]):
            perm = torch.randperm(batch)
            for i in range(0, batch, mb):
                idx = perm[i:i + mb]
                logits, v = policy(f_obs[idx], f_mask[idx])
                lp_all = F.log_softmax(logits, dim=-1)
                lp = lp_all.gather(1, f_act[idx].unsqueeze(1)).squeeze(1)
                ent = -(lp_all.exp() * lp_all).sum(-1).mean()
                A = f_adv[idx]
                if rc["adv_norm"]:
                    A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = (lp - f_logp[idx]).exp()
                pg = -torch.min(ratio * A, ratio.clamp(1 - rc["clip_eps"], 1 + rc["clip_eps"]) * A).mean()
                vloss = 0.5 * (v - f_ret[idx]).pow(2).mean()
                loss = pg - rc["ent_coef"] * ent + rc["vf_coef"] * vloss
                if not torch.isfinite(loss):
                    status = "diverged"
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), rc["max_grad_norm"])
                opt.step()
            if status != "ok":
                break
        if status != "ok":
            break
        curve.append({"update": upd, "steps": upd * batch, "train_exact": n_exact / max(n_end, 1),
                      "episodes": n_end})
        if upd in snap_at:
            snapshot(upd)
    tail = curve[-max(1, len(curve) // 10):] if curve else []
    torch.save({"snapshots": snaps, "world": spec.name, "candidate_sha": sha(src), "batch": batch,
                "n_updates": n_updates, "flops_ratio": flops_ratio}, out / "snapshots.pt")
    log = {"status": status, "seed": seed, "budget_steps": args.budget, "n_updates": n_updates, "batch": batch,
           "flops_ratio": round(flops_ratio, 4), "snap_at": snap_at, "wall_s": round(time.time() - t0, 1),
           "train_exact_tail": sum(c["train_exact"] for c in tail) / max(len(tail), 1), "curve": curve}
    (out / "train_log.json").write_text(json.dumps(log))
    print(json.dumps({k: v for k, v in log.items() if k != "curve"}))


if __name__ == "__main__":
    main()
