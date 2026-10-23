"""
Behavioral Cloning от Baseline (ai_common.Baseline): сбор пар (наблюдение, действие учителя)
на сценах методики ТЗ и случайных сценах (с потерей изображения, как в ai_env.py), обучение сети
повторять действие. Результат служит стартом для PPO
(ai_train.py) и якорем KL.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from ai_common import Baseline, device_of, encode, load_net, tensors
from ai_env import GliderEnv

def collect(a):
    env = GliderEnv(seed=a.seed, randomize=not a.default_scene, sudden_prob=0.0, max_steps=2500)
    teacher = Baseline()
    O, A = [], []
    for ep in range(a.episodes):
        obs, info = env.reset(), {}
        teacher.reset()
        for _ in range(env.max_steps):
            act = encode(*teacher(env.sim, env.s))
            O.append(obs); A.append(act)
            obs, _, done, info = env.step(act)
            if done:
                break
        print(f"[сбор] эпизод {ep + 1}/{a.episodes}: пар {len(O)}, цель={info.get('goal')}, "
              f"авария={info.get('reason') or '-'}")
    env.close()
    return np.stack(O), np.stack(A)

def train_bc(a):
    dev = device_of(a.cpu)
    obs, act = collect(a)
    net, _ = load_net(None, dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    nav, dep = tensors(obs, dev)
    tu = torch.as_tensor(act[:, :4], device=dev).clamp(-0.995, 0.995)   # руль высоты учитель не задает; ±1 tanh не достигает
    tem = torch.as_tensor((act[:, 5] > 0).astype(np.float32), device=dev)
    for ep in range(a.epochs):
        tot, n = 0.0, 0
        for mb in torch.randperm(len(obs), device=dev).split(a.minibatch):
            mean, logit, _, _, _ = net(nav[mb], dep[mb], net.initial_hidden(len(mb), dev))
            loss = F.mse_loss(torch.tanh(mean[:, :4]), tu[mb]) + 0.1 * F.binary_cross_entropy_with_logits(logit, tem[mb])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot, n = tot + float(loss.detach()), n + 1
        print(f"[BC] эпоха {ep + 1}/{a.epochs}: loss={tot / n:.5f}")
    torch.save(dict(model=net.state_dict()), a.out)
    print(f"[BC] сохранено: {a.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--default-scene", action="store_true", help="собирать только на сцене по умолчанию")
    ap.add_argument("--out", default="ai_pilot_bc.pth")
    ap.add_argument("--cpu", action="store_true")
    train_bc(ap.parse_args())
