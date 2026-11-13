"""Smoke test and rough timing: the frozen PPO training loop (arch_rsi/train_arch.py) ported to a device argument.

Imports the frozen modules unchanged and never writes inside the project. Same maths as train_arch.main;
the only change is that every buffer lives on `device`. Prints one JSON line per configuration.
usage: python -I bench_train.py <project_root> <candidate.py> <label> <device> <n_envs> <updates>
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arch_rsi.interface import CandidatePolicy, arch_spec  # noqa: E402
from arch_rsi.sandbox import load_candidate  # noqa: E402
from arch_rsi.train_arch import heldout_hashes  # noqa: E402
from arch_rsi.worlds import RECIPE, SPECS  # noqa: E402
from brickworld.env import BrickWorld, StreamSource  # noqa: E402
from brickworld.model import encode  # noqa: E402
from train import TRAIN_SEED_BASE, reward_fn  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def run(src, label, device, N, n_updates, seed=0):
    torch.set_num_threads(1)
    rc = dict(RECIPE)
    spec = SPECS["xl"]
    aspec = arch_spec(spec)
    torch.manual_seed(seed)
    policy = CandidatePolicy(load_candidate(src)(aspec), aspec).to(device)
    stream = StreamSource(spec, TRAIN_SEED_BASE + seed, heldout_hashes(spec.name))
    T = rc["rollout_len"]
    world = BrickWorld(spec, N, device, target_source=stream)
    opt = torch.optim.Adam(policy.parameters(), lr=rc["lr"], eps=1e-5)
    batch = N * T
    mb = batch // rc["n_minibatches"]
    obs = encode(world, "voxels")
    C = obs.shape[1]
    z = lambda *s, **k: torch.zeros(*s, device=device, **k)  # noqa: E731
    b_obs, b_mask = z(T, N, C, spec.Y, spec.X), z(T, N, spec.n_actions, dtype=torch.bool)
    b_act = z(T, N, dtype=torch.long)
    b_logp, b_val, b_rew, b_done = (z(T, N) for _ in range(4))
    ep_ret = z(N)
    times, losses = [], []
    for upd in range(n_updates + 1):  # update 0 is warm-up, not timed
        sync(device)
        t0 = time.perf_counter()
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
                ep_ret[ended] = 0.0
            obs = encode(world, "voxels")
        with torch.no_grad():
            _, next_val = policy(obs, world.valid_mask())
        adv, last = z(T, N), z(N)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            nv = next_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + rc["gamma"] * nv * nonterm - b_val[t]
            last = delta + rc["gamma"] * rc["gae_lambda"] * nonterm * last
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
                A = (A - A.mean()) / (A.std() + 1e-8)
                ratio = (lp - f_logp[idx]).exp()
                pg = -torch.min(ratio * A, ratio.clamp(1 - rc["clip_eps"], 1 + rc["clip_eps"]) * A).mean()
                vloss = 0.5 * (v - f_ret[idx]).pow(2).mean()
                loss = pg - rc["ent_coef"] * ent + rc["vf_coef"] * vloss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), rc["max_grad_norm"])
                opt.step()
        sync(device)
        if upd > 0:
            times.append(time.perf_counter() - t0)
            losses.append(float(loss))
    s_per_upd = sorted(times)[len(times) // 2]
    return {"label": label, "device": device.type, "n_envs": N, "updates_timed": n_updates,
            "s_per_update": round(s_per_upd, 3), "steps_per_s": round(batch / s_per_upd),
            "finite": all(map(lambda x: x == x and abs(x) < 1e9, losses))}


def parity(src):
    """Same weights, same input: CPU vs MPS forward outputs."""
    spec = SPECS["xl"]
    aspec = arch_spec(spec)
    torch.manual_seed(0)
    net = CandidatePolicy(load_candidate(src)(aspec), aspec)
    world = BrickWorld(spec, 256, "cpu", target_source=StreamSource(spec, 999, heldout_hashes("xl")))
    obs, mask = encode(world, "voxels"), world.valid_mask()
    with torch.no_grad():
        lc, vc = net(obs, mask)
        net_m = net.to("mps")
        lm, vm = net_m(obs.to("mps"), mask.to("mps"))
    legal = mask
    return {"max_logit_diff": float((lc - lm.cpu())[legal].abs().max()), "max_value_diff": float((vc - vm.cpu()).abs().max())}


if __name__ == "__main__":
    cand, label, dev, N, U = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
    src = Path(cand).read_text()
    if dev == "parity":
        print(json.dumps({"label": label, **parity(src)}))
    else:
        print(json.dumps(run(src, label, torch.device(dev), N, U)))
