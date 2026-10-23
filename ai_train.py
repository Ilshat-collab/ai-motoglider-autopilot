"""
PPO с рекуррентной сетью (GRU) на сценах методики ТЗ, случайных сценах, с внезапными препятствиями
и потерей изображения (см. ai_env.py). Старт с весов --init (если нет, с BC-весов --bc-kl-model),
мягкий KL-якорь к BC-политике. Веса сохраняются в .pth.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from ai_common import device_of, load_net, tensors
from ai_env import GliderEnv

def gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    adv, last = np.zeros(len(rew), dtype=np.float32), 0.0
    nxt = np.append(val[1:], last_val)
    for t in reversed(range(len(rew))):
        nd = 1.0 - done[t]
        last = rew[t] + gamma * nxt[t] * nd - val[t] + gamma * lam * nd * last
        adv[t] = last
    return adv, adv + val

def kl_to_bc(mean, log_std, logit, mean_bc, log_std_bc, logit_bc):
    kl_c = (log_std_bc - log_std + ((2 * log_std).exp() + (mean - mean_bc) ** 2) / (2 * (2 * log_std_bc).exp())
            - 0.5).sum(-1).mean()
    p, q = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6), torch.sigmoid(logit_bc).clamp(1e-6, 1 - 1e-6)
    return kl_c + (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()

def entropy(log_std, logit):
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    return (0.5 * math.log(2 * math.pi * math.e) + log_std).sum(-1).mean() - (p * p.log() + (1 - p) * (1 - p).log()).mean()

def train(a):
    dev = device_of(a.cpu)
    env = GliderEnv(seed=a.seed, sudden_prob=a.sudden, route_prob=a.route, blackout_prob=a.blackout)
    net, ok = load_net(None if a.no_init else a.init, dev)
    if not ok and not a.no_init:
        net, ok = load_net(a.bc_kl_model, dev)
    print(f"[init] {'веса загружены' if ok else 'обучение с нуля'}")
    bc = None
    if a.bc_kl > 0:
        bc, ok = load_net(a.bc_kl_model, dev)
        bc = bc.eval().requires_grad_(False) if ok else None
        print(f"[bc-kl] {'коэф ' + str(a.bc_kl) if bc else 'файл не найден, KL отключен'}")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    obs, h = env.reset(), net.initial_hidden(1, dev)
    hb = net.initial_hidden(1, dev)
    print(f"Старт: updates={a.updates}, rollout={a.rollout}, device={dev}, lr={a.lr}")

    for upd in range(1, a.updates + 1):
        env.em_bonus = a.em_bonus if upd <= a.em_bonus_updates else 0.0
        O, H, HB, A, LP, V, R, Dn = [], [], [], [], [], [], [], []
        st = dict(win=0, crash=0, ep=0, em=0, fb=0, over=0)
        for _ in range(a.rollout):
            nav, dep = tensors(obs, dev)
            with torch.no_grad():
                act, lp, v, h_next = net.act(nav, dep, h)
                hb_next = bc(nav, dep, hb)[4] if bc is not None else hb
            ac = act[0].cpu().numpy().astype(np.float32)
            O.append(obs); H.append(h); HB.append(hb); A.append(ac); LP.append(float(lp[0])); V.append(float(v[0]))
            obs, r, done, info = env.step(ac)
            R.append(r); Dn.append(float(done))
            st["em"] += info["emergency"]; st["fb"] += info["fallback"]; st["over"] += info["overfly"]
            if done:
                st["ep"] += 1; st["win"] += info["goal"]; st["crash"] += info["crashed"]
                obs, h, hb = env.reset(), net.initial_hidden(1, dev), net.initial_hidden(1, dev)
            else:
                h, hb = h_next, hb_next
        with torch.no_grad():
            last_v = 0.0 if Dn[-1] else float(net.act(*tensors(obs, dev), h)[2][0])
        adv, ret = gae(np.array(R, np.float32), np.array(V, np.float32), np.array(Dn, np.float32), last_v)

        nav_t, dep_t = tensors(np.stack(O), dev)
        h_t, hb_t = torch.cat(H), torch.cat(HB)
        act_t, lp_t = torch.as_tensor(np.stack(A), device=dev), torch.as_tensor(LP, device=dev)
        adv_t = torch.as_tensor(adv, device=dev)
        adv_t = ((adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)).clamp(-3, 3)
        ret_t = torch.as_tensor(ret, device=dev)
        acc = np.zeros(3)
        n = 0
        for _ in range(a.epochs):
            for mb in torch.randperm(a.rollout, device=dev).split(a.minibatch):
                mean, logit, log_std, value, _ = net(nav_t[mb], dep_t[mb], h_t[mb])
                ratio = (net.recompute_logp(act_t[mb], mean, log_std, logit) - lp_t[mb]).exp()
                pi = -torch.min(ratio * adv_t[mb], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[mb]).mean()
                vl = ((value - ret_t[mb]) ** 2).mean()
                loss = pi + 0.5 * vl - a.ent * entropy(log_std, logit)
                kl = torch.zeros((), device=dev)
                if bc is not None:
                    with torch.no_grad():
                        mean_b, logit_b, log_std_b, _, _ = bc(nav_t[mb], dep_t[mb], hb_t[mb])
                    kl = kl_to_bc(mean, log_std, logit, mean_b, log_std_b, logit_b)
                    loss = loss + a.bc_kl * kl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                acc += [float(pi), float(vl), float(kl)]
                n += 1
        acc /= max(n, 1)
        k = 100.0 / a.rollout
        print(f"[{upd:4d}/{a.updates}] pi={acc[0]:.3f} v={acc[1]:.3f} kl={acc[2]:.4f} эпизодов={st['ep']} "
              f"успех={st['win']} аварий={st['crash']} экстр={st['em'] * k:.1f}% fallback={st['fb'] * k:.1f}% "
              f"над_препятствием={st['over'] * k:.1f}% бонус={env.em_bonus}")
        if upd % a.save_every == 0 or upd == a.updates:
            torch.save(dict(model=net.state_dict(), args=vars(a)), a.out)
            print(f"  сохранено: {a.out}")
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--rollout", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--sudden", type=float, default=0.3, help="вероятность внезапного препятствия")
    ap.add_argument("--route", type=float, default=0.5, help="доля сцен методики ТЗ")
    ap.add_argument("--blackout", type=float, default=0.2, help="доля эпизодов с потерей изображения")
    ap.add_argument("--em-bonus", type=float, default=0.0, help="временный бонус за экстренный режим у препятствия")
    ap.add_argument("--em-bonus-updates", type=int, default=50)
    ap.add_argument("--bc-kl", type=float, default=0.005)
    ap.add_argument("--bc-kl-model", default="ai_pilot_bc.pth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", default="ai_pilot.pth")
    ap.add_argument("--init", default="ai_pilot.pth")
    ap.add_argument("--no-init", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    train(ap.parse_args())
