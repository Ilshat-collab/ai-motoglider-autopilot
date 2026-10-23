"""
Рекуррентный Actor-Critic: сверточный кодер карты глубины 40x64 + 26 навигационных признаков,
два полносвязных слоя, GRUCell 128. Выход актора: 5 средних непрерывных действий до tanh
(крен, скорость, скольжение, газ, руль высоты) и логит экстренного режима. log_std общий.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_MAX = 4.5
LOG2PI = math.log(2 * math.pi)

class DepthEncoder(nn.Module):
    def __init__(self, out_dim=64, depth_shape=(40, 64)):
        super().__init__()
        h, w = (math.ceil(x / 8) for x in depth_shape)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * h * w, out_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, nav_dim=26, depth_shape=(40, 64), hidden=128, use_depth=True):
        super().__init__()
        self.use_depth, self.hidden = use_depth, hidden
        if use_depth:
            self.depth_enc = DepthEncoder(64, depth_shape)
        self.body = nn.Sequential(nn.Linear(nav_dim + 64 * use_depth, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor = nn.Linear(hidden, 6)
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((5,), -1.5))
        with torch.no_grad():
            # крен 0, скорость ~12 м/с, скольжение 0, газ ~27 %, руль высоты -7 град (кабрирование), экстр ~5 %
            self.actor.bias.copy_(torch.tensor([0.0, -0.576, 0.0, -0.5, -0.510, -3.0]))
            self.critic.weight.mul_(0.1)
            self.critic.bias.zero_()

    def initial_hidden(self, batch=1, device=None):
        return torch.zeros(batch, self.hidden, device=device or self.log_std.device)

    def forward(self, nav, depth, h):
        x = torch.cat([nav, self.depth_enc(depth)], -1) if self.use_depth else nav
        h = self.gru(self.body(x), h)
        out = self.actor(h)
        log_std = self.log_std.clamp(-3.0, 0.5).expand(out.size(0), -1)
        return out[:, :5], out[:, 5], log_std, self.critic(h).squeeze(-1), h

    @staticmethod
    def _logp(x, mean, log_std, logit, em):
        """log-плотность squashed Gaussian в пространстве u = tanh(x) плюс Бернулли экстренного режима."""
        u = torch.tanh(x)
        lp = (-0.5 * ((x - mean) / log_std.exp()) ** 2 - log_std - 0.5 * LOG2PI - torch.log(1 - u * u + 1e-6)).sum(-1)
        return lp + F.logsigmoid(logit) * em + F.logsigmoid(-logit) * (1 - em), u

    def act(self, nav, depth, h, deterministic=False):
        mean, logit, log_std, value, h = self(nav, depth, h)
        if deterministic:
            x, em = mean, (logit > 0).float()
        else:
            x, em = mean + log_std.exp() * torch.randn_like(mean), torch.bernoulli(torch.sigmoid(logit))
        logp, u = self._logp(x.clamp(-RAW_MAX, RAW_MAX), mean, log_std, logit, em)
        return torch.cat([u, em.unsqueeze(-1)], -1), logp, value, h

    def recompute_logp(self, act, mean, log_std, logit):
        lim = math.tanh(RAW_MAX)
        x = torch.atanh(act[:, :5].clamp(-lim, lim))
        return self._logp(x, mean, log_std, logit, act[:, 5])[0]

class _Deploy(nn.Module):
    """Детерминированная политика для борта: действие [-1, 1]^5, вероятность экстренного режима, новое h."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, nav, depth, h):
        mean, logit, _, _, h = self.net(nav, depth, h)
        return torch.tanh(mean.clamp(-RAW_MAX, RAW_MAX)), torch.sigmoid(logit), h

def export_onnx(weights, out, nav_dim=26, depth_shape=(40, 64)):
    net = ActorCritic(nav_dim, depth_shape)
    ck = torch.load(weights, map_location="cpu")
    net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    dep = _Deploy(net.eval())
    args = (torch.zeros(1, nav_dim), torch.zeros(1, 1, *depth_shape), net.initial_hidden(1))
    torch.onnx.export(dep, args, out, input_names=["nav", "depth", "h"], output_names=["action", "p_emergency", "h_out"],
                      dynamic_axes={k: {0: "batch"} for k in ("nav", "depth", "h", "action", "p_emergency", "h_out")},
                      opset_version=17)
    with torch.no_grad():
        for _ in range(20):
            dep(*args)
        t0 = time.perf_counter()
        for _ in range(200):
            dep(*args)
        ms = (time.perf_counter() - t0) / 200 * 1000
    n = sum(p.numel() for p in net.parameters())
    print(f"[onnx] {out}: {os.path.getsize(out) / 1e6:.2f} МБ (норма ТЗ <= 1000 МБ), параметров {n}, "
          f"инференс на CPU {ms:.2f} мс на кадр (норма ТЗ <= 30 мс, частота >= 20 Гц)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ai_pilot.pth")
    ap.add_argument("--onnx", default="ai_pilot.onnx")
    a = ap.parse_args()
    export_onnx(a.weights, a.onnx)
