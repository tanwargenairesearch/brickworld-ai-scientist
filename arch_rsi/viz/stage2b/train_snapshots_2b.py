"""VISUALIZATION ONLY, never used for any score, selection or analysis (Stage 2b video, scene 1).

A line-for-line copy of arch_rsi/stage2/train_2b.py (same RECIPE handling, fixed-FLOP budget, reward, design stream,
W's held-out rejection, seed and device). The only additions: it keeps copies of the policy weights at 11 evenly
spaced points in training (update 0 = untrained, then every 10% of the updates) and saves them to snapshots.pt.
Copying weights draws no random numbers, so training is unchanged; the renderer checks the final snapshot against
the Stage 2b confirmation run's own dev score and stops if they differ.

usage (project root): .venv/bin/python -I arch_rsi/viz/stage2b/train_snapshots_2b.py --spec run.json --out DIR
run.json: as for train_2b.py: {"candidate", "world", "seed", "budget_steps", "device", "threads"}.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate, sha  # noqa: E402
from arch_rsi.train_arch import heldout_hashes  # noqa: E402
from arch_rsi.worlds import RECIPE, SPECS  # noqa: E402
from arch_rsi.stage2 import w2b, w_world  # noqa: E402
from brickworld.env import BrickWorld, StreamSource  # noqa: E402
from brickworld.model import encode  # noqa: E402
from train import TRAIN_SEED_BASE, reward_fn  # noqa: E402

BASELINE = ROOT / "arch_rsi" / "candidates" / "baseline.py"
w_world.register()


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


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
    try:
        file_recipe = w2b.parse_recipe(src)
        rerr = w2b.validate(file_recipe)
    except (ValueError, SyntaxError) as e:
        file_recipe, rerr = None, [str(e)]
    if rerr:
        log = {"status": "invalid", "error": "; ".join(rerr)[:500], "wall_s": 0.0}
        (out / "train_log.json").write_text(json.dumps(log))
        print(json.dumps(log))
        return
    rc = {**RECIPE, **w2b.full_recipe(file_recipe)}
    spec = SPECS[run["world"]]
    aspec = arch_spec(spec)
    seed = int(run["seed"])
    device = torch.device(run.get("device", "mps"))
    torch.set_num_threads(int(run.get("threads", 1)))
    torch.manual_seed(seed)
    t0 = time.time()
    curve, status, error = [], "ok", None
    try:
        from arch_rsi.check import train_flops
        flops_ratio = train_flops(src, aspec) / train_flops(BASELINE.read_text(), aspec)
        torch.manual_seed(seed)
        build = load_candidate(src)
        policy = CandidatePolicy(build(aspec), aspec).to(device)
    except Exception as e:  # noqa: BLE001
        log = {"status": "invalid", "error": f"{type(e).__name__}: {str(e)[:500]}", "wall_s": 0.0}
        (out / "train_log.json").write_text(json.dumps(log))
        print(json.dumps(log))
        return

    bud = w2b.budget(int(run["budget_steps"]), flops_ratio, file_recipe)
    if not bud["ok"]:
        log = {"status": "invalid", "error": f"the budget allows only {bud['n_updates']} updates of {bud['batch']} steps; "
                                             f"at least {w2b.MIN_UPDATES} are required", "wall_s": 0.0}
        (out / "train_log.json").write_text(json.dumps(log))
        print(json.dumps(log))
        return
    hh = w_world.w_hashes() if spec.name == "w" else heldout_hashes(spec.name)
    stream = StreamSource(spec, TRAIN_SEED_BASE + seed, hh)
    N, T = rc["n_envs"], rc["rollout_len"]
    world = BrickWorld(spec, N, device, target_source=stream)
    opt = torch.optim.Adam(policy.parameters(), lr=rc["lr"], eps=1e-5)
    n_updates = bud["n_updates"]
    N_SNAP = 10  # visualization only: snapshots at k/N_SNAP of the updates
    snap_at = sorted({round(k * n_updates / N_SNAP) for k in range(N_SNAP + 1)})
    snaps = {}

    def snapshot(u):  # a copy of the weights; no random numbers drawn
        snaps[u] = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    snapshot(0)
    batch = N * T
    mb = batch // rc["n_minibatches"]
    obs = encode(world, "voxels")
    C = obs.shape[1]

    def z(*s, **k):
        return torch.zeros(*s, device=device, **k)

    b_obs = z(T, N, C, spec.Y, spec.X)
    b_mask = z(T, N, spec.n_actions, dtype=torch.bool)
    b_act = z(T, N, dtype=torch.long)
    b_logp, b_val, b_rew, b_done = (z(T, N) for _ in range(4))
    ep_ret = z(N)
    meta = {"params": sum(p.numel() for p in policy.parameters()), "n_updates": n_updates,
            "flops_ratio": round(flops_ratio, 4), "budget_steps": int(run["budget_steps"]), "candidate_sha": sha(src),
            "device": device.type, "recipe": {k: rc[k] for k in w2b.BOUNDS}, "recipe_from_file": file_recipe,
            "allowed_steps": bud["allowed_steps"]}
    loop_t0 = time.time()
    try:
        for upd in range(1, n_updates + 1):
            if rc["lr_schedule"] == "linear_decay":
                for g in opt.param_groups:
                    g["lr"] = rc["lr"] * (1.0 - (upd - 1) / n_updates)
            n_end, n_exact, len_sum, ret_sum = z(()), z(()), z(()), z(())
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
                endf = ended.float()
                b_done[t] = endf
                ep_ret += r
                n_end += endf.sum()
                n_exact += (info["exact"] & ended).float().sum()
                len_sum += (info["ep_len"].float() * endf).sum()
                ret_sum += (ep_ret * endf).sum()
                ep_ret = torch.where(ended, torch.zeros_like(ep_ret), ep_ret)
                obs = encode(world, "voxels")
            with torch.no_grad():
                _, next_val = policy(obs, world.valid_mask())
            adv, last = z(T, N), z(N)
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
                perm = torch.randperm(batch, device=device)
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
            ne, nx, ls, rs = (float(x) for x in (n_end, n_exact, len_sum, ret_sum))
            curve.append({"update": upd, "steps": upd * batch, "train_exact": nx / max(ne, 1),
                          "ep_len": ls / max(ne, 1), "ep_return": rs / max(ne, 1), "episodes": int(ne)})
            if upd in snap_at:
                snapshot(upd)
        sync(device)
    except Exception as e:  # noqa: BLE001
        status, error = "crashed", f"{type(e).__name__}: {str(e)[:400]} | {traceback.format_exc()[-600:]}"
    loop_t1 = time.time()
    wall = loop_t1 - t0
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    torch.save({"state_dict": state, "world": spec.name, "candidate_sha": sha(src),
                "steps": len(curve) * batch}, out / "ckpt.pt")
    torch.save({"snapshots": {u: {k: v.cpu() for k, v in sd.items()} for u, sd in snaps.items()},
                "world": spec.name, "candidate_sha": sha(src), "batch": batch, "n_updates": n_updates,
                "flops_ratio": flops_ratio, "snap_at": snap_at}, out / "snapshots.pt")
    tail = curve[-max(1, len(curve) // 10):] if curve else []
    log = {"status": status, "error": error, "wall_s": round(wall, 2), "steps": len(curve) * batch,
           "sps": round(len(curve) * batch / max(wall, 1e-9), 1), **meta,
           "loop_t0": loop_t0, "loop_t1": loop_t1, "loop_sps": round(len(curve) * batch / max(loop_t1 - loop_t0, 1e-9), 1),
           "train_exact_tail": sum(c["train_exact"] for c in tail) / max(len(tail), 1),
           "designs_drawn": stream.drawn, "designs_rejected_heldout": stream.rejected,
           "snap_at": snap_at, "curve": curve[:: max(1, len(curve) // 100)]}
    (out / "train_log.json").write_text(json.dumps(log))
    print(json.dumps({k: v for k, v in log.items() if k != "curve"}))


if __name__ == "__main__":
    main()
