"""Train one PPO agent from a run spec. Frozen code: proposers only change the recipe JSON.

usage: python train.py --spec run.json --out DIR
run.json: {"recipe": {...search-space knobs...}, "scale": "small", "seed": 0,
           "budget_steps": 1000000, "train_targets": "stream" | "fixed:<N>",
           "ckpt_fracs": [], "device": "cpu", "threads": 2, "width_mult": 1}
The trial never scores itself on held-out designs: evaluate.py does that in a separate process.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from brickworld.env import SCALES, BrickWorld, FixedSource, StreamSource
from brickworld.model import Policy, encode
from brickworld.space import validate

ROOT = Path(__file__).resolve().parent
TRAIN_SEED_BASE = 10_000_000  # training-design streams use seeds far from the held-out sets


def reward_fn(rc: dict, info: dict, placed: torch.Tensor) -> torch.Tensor:
    mode, bonus = rc["reward_mode"], rc["success_bonus"]
    exact = info["exact"].float()
    if mode == "sparse":
        r = (1.0 + bonus) * exact
    elif mode == "cells":
        r = (info["cells_in"] - info["cells_out"]).float() / info["target_cells"].clamp(min=1).float()
        r = r + bonus * exact
    elif mode == "iou_delta":
        r = info["iou_new"] - info["iou_prev"] + bonus * exact
    elif mode == "heightmap_delta":
        r = info["hm_new"] - info["hm_prev"] + bonus * info["hm_match"].float()
    else:
        raise ValueError(mode)
    return r - rc["step_penalty"] * placed.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = validate(run["recipe"])
    spec = SCALES[run["scale"]]
    seed = int(run["seed"])
    device = torch.device(run.get("device", "cpu"))
    torch.set_num_threads(int(run.get("threads", 2)))
    torch.manual_seed(seed)
    net_cfg = dict(rc)
    net_cfg["width"] = int(rc["width"] * run.get("width_mult", 1))

    heldout = torch.load(ROOT / "targets" / f"{spec.name}_heldout_hashes.pt")
    stream = StreamSource(spec, TRAIN_SEED_BASE + seed, heldout)
    tt = run.get("train_targets", "stream")
    if tt == "stream":
        source = stream
    else:
        n_fixed = int(tt.split(":")[1])
        designs = stream(n_fixed).clone()
        torch.save(designs, out / "train_designs.pt")
        source = FixedSource(designs, seed)

    N, T = rc["n_envs"], rc["rollout_len"]
    world = BrickWorld(spec, N, device, target_source=source)
    policy = Policy(spec, net_cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=rc["lr"], eps=1e-5)
    n_updates = max(1, int(run["budget_steps"]) // (N * T))
    batch = N * T
    mb = batch // rc["n_minibatches"]
    ckpt_at = {max(1, round(f * n_updates)): f for f in run.get("ckpt_fracs", [])}

    obs_mode = rc["obs_mode"]
    obs = encode(world, obs_mode)
    C = obs.shape[1]
    b_obs = torch.zeros(T, N, C, spec.Y, spec.X, device=device)
    b_mask = torch.zeros(T, N, spec.n_actions, dtype=torch.bool, device=device)
    b_act = torch.zeros(T, N, dtype=torch.long, device=device)
    b_logp = torch.zeros(T, N, device=device)
    b_val = torch.zeros(T, N, device=device)
    b_rew = torch.zeros(T, N, device=device)
    b_done = torch.zeros(T, N, device=device)

    t0 = time.time()
    curve, status = [], "ok"
    ep_ret = torch.zeros(N, device=device)
    meta = {"params": sum(p.numel() for p in policy.parameters()), "n_updates": n_updates}
    for upd in range(1, n_updates + 1):
        if rc["lr_schedule"] == "linear_decay":
            for g in opt.param_groups:
                g["lr"] = rc["lr"] * (1.0 - (upd - 1) / n_updates)
        n_end = n_exact = n_hm = 0
        len_sum = ret_sum = 0.0
        for t in range(T):
            mask = world.valid_mask()
            with torch.no_grad():
                logits, value = policy(obs, mask)
                probs = F.softmax(logits, dim=-1)
                a = torch.multinomial(probs, 1).squeeze(1)
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
                n_hm += int(info["hm_match"].sum())
                len_sum += float(info["ep_len"][ended].float().sum())
                ret_sum += float(ep_ret[ended].sum())
                ep_ret[ended] = 0.0
            obs = encode(world, obs_mode)
        with torch.no_grad():
            _, next_val = policy(obs, world.valid_mask())
        adv = torch.zeros(T, N, device=device)
        last = torch.zeros(N, device=device)
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
        curve.append({
            "update": upd, "steps": upd * batch,
            "train_exact": n_exact / max(n_end, 1), "train_hm": n_hm / max(n_end, 1),
            "ep_len": len_sum / max(n_end, 1), "ep_return": ret_sum / max(n_end, 1),
            "episodes": n_end,
        })
        if upd in ckpt_at:
            torch.save({"state_dict": policy.state_dict(), "net_cfg": net_cfg, "scale": spec.name,
                        "steps": upd * batch}, out / f"ckpt_{ckpt_at[upd]:.3f}.pt")

    wall = time.time() - t0
    torch.save({"state_dict": policy.state_dict(), "net_cfg": net_cfg, "scale": spec.name,
                "steps": len(curve) * batch}, out / "ckpt.pt")
    tail = curve[-max(1, len(curve) // 10):] if curve else []
    log = {
        "status": status, "wall_s": round(wall, 2), "steps": len(curve) * batch,
        "sps": round(len(curve) * batch / max(wall, 1e-9), 1), **meta,
        "train_exact_tail": sum(c["train_exact"] for c in tail) / max(len(tail), 1),
        "train_hm_tail": sum(c["train_hm"] for c in tail) / max(len(tail), 1),
        "designs_drawn": stream.drawn, "designs_rejected_heldout": stream.rejected,
        "curve": curve[:: max(1, len(curve) // 200)],
    }
    (out / "train_log.json").write_text(json.dumps(log))
    print(json.dumps({k: v for k, v in log.items() if k != "curve"}))


if __name__ == "__main__":
    main()
