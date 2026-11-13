"""Train one PPO agent whose network comes from a candidate architecture file. Frozen code.

usage: python -I -m arch_rsi.train_arch --spec run.json --out DIR      (run from the project root)
run.json: {"candidate": "path/to/candidate.py", "world": "large"|"xl", "seed": 0,
           "budget_steps": 524288, "threads": 1}
Compute-matched budget: budget_steps is the number of environment steps the starting architecture trains
for. A candidate whose training FLOPs per sample are r times the starting architecture's (counted by
check.train_flops) trains for budget_steps / r steps, rounded down to whole updates; a cheaper network still
gets budget_steps. Everything except the network is fixed: the PPO recipe (worlds.RECIPE), the observation,
the reward, the compute budget and the design stream. The trial never scores itself on held-out designs.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.worlds import RECIPE, SPECS  # noqa: E402
from brickworld.env import BrickWorld, StreamSource  # noqa: E402
from brickworld.model import encode  # noqa: E402
from train import TRAIN_SEED_BASE, reward_fn  # noqa: E402

HERE = Path(__file__).resolve().parent


def heldout_hashes(world: str) -> torch.Tensor:
    if world == "xl":
        return torch.load(HERE / "targets" / "xl_heldout_hashes.pt")
    return torch.load(ROOT / "targets" / f"{world}_heldout_hashes.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(run["candidate"]).read_text()
    (out / "candidate.py").write_text(src)
    rc = dict(RECIPE)
    spec = SPECS[run["world"]]
    aspec = arch_spec(spec)
    seed = int(run["seed"])
    device = torch.device("cpu")
    torch.set_num_threads(int(run.get("threads", 1)))
    torch.manual_seed(seed)
    t0 = time.time()
    curve, status, error = [], "ok", None
    try:
        from arch_rsi.check import train_flops
        flops_ratio = train_flops(src, aspec) / train_flops((HERE / "candidates" / "baseline.py").read_text(), aspec)
        torch.manual_seed(seed)
        build = load_candidate(src)
        policy = CandidatePolicy(build(aspec), aspec).to(device)
    except Exception as e:  # noqa: BLE001
        log = {"status": "invalid", "error": f"{type(e).__name__}: {str(e)[:500]}", "wall_s": 0.0}
        (out / "train_log.json").write_text(json.dumps(log))
        print(json.dumps(log))
        return

    stream = StreamSource(spec, TRAIN_SEED_BASE + seed, heldout_hashes(spec.name))
    N, T = rc["n_envs"], rc["rollout_len"]
    world = BrickWorld(spec, N, device, target_source=stream)
    opt = torch.optim.Adam(policy.parameters(), lr=rc["lr"], eps=1e-5)
    base_updates = max(1, int(run["budget_steps"]) // (N * T))
    n_updates = max(1, int(base_updates * min(1.0, 1.0 / flops_ratio)))
    batch = N * T
    mb = batch // rc["n_minibatches"]
    obs = encode(world, "voxels")
    C = obs.shape[1]
    b_obs = torch.zeros(T, N, C, spec.Y, spec.X)
    b_mask = torch.zeros(T, N, spec.n_actions, dtype=torch.bool)
    b_act = torch.zeros(T, N, dtype=torch.long)
    b_logp, b_val, b_rew, b_done = (torch.zeros(T, N) for _ in range(4))
    ep_ret = torch.zeros(N)
    meta = {"params": sum(p.numel() for p in policy.parameters()), "n_updates": n_updates,
            "flops_ratio": round(flops_ratio, 4), "budget_steps": int(run["budget_steps"]), "candidate_sha": sha(src)}
    try:
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
                          "ep_len": len_sum / max(n_end, 1), "ep_return": ret_sum / max(n_end, 1),
                          "episodes": n_end})
    except Exception as e:  # noqa: BLE001
        status, error = "crashed", f"{type(e).__name__}: {str(e)[:400]} | {traceback.format_exc()[-600:]}"
    wall = time.time() - t0
    torch.save({"state_dict": policy.state_dict(), "world": spec.name, "candidate_sha": sha(src),
                "steps": len(curve) * batch}, out / "ckpt.pt")
    tail = curve[-max(1, len(curve) // 10):] if curve else []
    log = {"status": status, "error": error, "wall_s": round(wall, 2), "steps": len(curve) * batch,
           "sps": round(len(curve) * batch / max(wall, 1e-9), 1), **meta,
           "train_exact_tail": sum(c["train_exact"] for c in tail) / max(len(tail), 1),
           "designs_drawn": stream.drawn, "designs_rejected_heldout": stream.rejected,
           "curve": curve[:: max(1, len(curve) // 100)]}
    (out / "train_log.json").write_text(json.dumps(log))
    print(json.dumps({k: v for k, v in log.items() if k != "curve"}))


if __name__ == "__main__":
    main()
