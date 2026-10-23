"""
Среда обучения с подкреплением поверх motoglider.GliderSim. Сеть видит то же, что на борту:
навигационные признаки и карту глубины через ai_common.Perception (с прогнозом по IMU при потере кадра);
уставки проходят через ai_common.Executor, как в оценке и демонстрации.
Добавлено (без этого сеть не держит высоту цели и учится перепрыгивать препятствия):
  отклонение от высоты цели   до -0.05 в секунду (линейно от 10 до 40 м отклонения)
  пролет над препятствием     -0.5 в секунду, если его выполняет сеть, а не предохранитель
  временный бонус за экстренный режим у препятствия (< 40 м), задается из ai_train.py, по умолчанию 0.

Сцены: с вероятностью route_prob сцена методики ТЗ (одно препятствие из 5 типов на маршруте на высоте
цели), иначе случайная сцена симулятора; цель на 40-120 м, старт около высоты цели, ветер 0-8 м/с,
внезапное препятствие на 25-50 м с вероятностью sudden_prob, потеря изображения на 0,2-1 с
с вероятностью blackout_prob.
"""

import numpy as np
import motoglider as mg
from ai_common import (ACT_DIM, DEPTH_SHAPE, DT, GOAL_RADIUS, NAV_DIM, Executor, Perception, decode, observe,
                       obstacle_info, route_scene)

R_APPROACH, R_GOAL, R_CRASH = 0.01, 10.0, -10.0
R_ENERGY_KJ, R_MOTOR_ON = -0.05, -0.2
R_LOW_ALT, LOW_ALT = -0.2, 30.0
R_NEAR_OBS, NEAR_OBS_DIST = -0.5, 15.0
R_SHARP, R_TIME, R_EMERGENCY = -0.02, -0.01, -0.1
R_ALT_HOLD, ALT_TOL, ALT_SPAN = -0.05, 10.0, 30.0
R_OVERFLY = -0.5
EM_BONUS_DIST = 40.0

GOAL_Z, WIND_MAX = (40.0, 120.0), 8.0
SUDDEN_DIST = (25.0, 50.0)

def reward(prev, s, act, prev_act, d_obs, over, em_net, fb, em_bonus=0.0):
    r = R_APPROACH * (prev["goal_dist"] - s["goal_dist"]) + R_TIME * DT
    r += R_GOAL * (s["goal_dist"] < GOAL_RADIUS) + R_CRASH * bool(s["crashed"])
    r += R_ENERGY_KJ * max(0.0, s["E_motor"] - prev["E_motor"]) / 1000.0
    r += R_MOTOR_ON * (s["motor_on"] and not prev["motor_on"])
    r += R_LOW_ALT * DT * (s["H"] < LOW_ALT) + R_NEAR_OBS * DT * (d_obs < NEAR_OBS_DIST)
    r += R_SHARP * float(np.sum((act[:4] - prev_act[:4]) ** 2))
    r += R_EMERGENCY * DT * (em_net or fb)
    r += em_bonus * DT * (em_net and not fb and d_obs < EM_BONUS_DIST)
    r += R_ALT_HOLD * DT * min(1.0, max(0.0, abs(s["H"] - s["goal"][2]) - ALT_TOL) / ALT_SPAN)
    r += R_OVERFLY * DT * (over and not fb)
    return float(r)

class GliderEnv:
    nav_dim, depth_shape, act_dim = NAV_DIM, DEPTH_SHAPE, ACT_DIM

    def __init__(self, seed=0, randomize=True, sudden_prob=0.3, route_prob=0.5, blackout_prob=0.2, max_steps=2000,
                 fallback=True, em_bonus=0.0):
        self.sim = mg.GliderSim(seed=seed, depth=True)
        self.rng = np.random.default_rng(seed)
        self.randomize, self.sudden_prob, self.route_prob, self.blackout_prob = randomize, sudden_prob, route_prob, blackout_prob
        self.max_steps, self.em_bonus = max_steps, em_bonus
        self.per, self.exe = Perception(), Executor(fallback)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sim, rng = self.sim, self.rng
        wind = (rng.uniform(0, WIND_MAX), rng.uniform(0, 360))
        zg = rng.uniform(*GOAL_Z)
        if not self.randomize:
            sim.default_scene()
            s = sim.reset()
        elif rng.random() < self.route_prob:
            s, _ = route_scene(sim, rng, H=zg, L=rng.uniform(500, 900), V=rng.uniform(11, 16), wind=wind,
                               bearing=rng.uniform(0, 360))
        else:
            start, goal, hdg = sim.randomize_scene(rng, start=(0.0, 0.0, zg + rng.uniform(-10, 40)))
            s = sim.reset(start=start, heading_deg=hdg, airspeed=rng.uniform(10, 16), wind=wind,
                          goal=(goal[0], goal[1], zg), randomize_obstacles=True)
        if rng.random() < self.sudden_prob:
            sim.spawn_obstacle(rng.uniform(*SUDDEN_DIST))
        self.dark = set()
        if rng.random() < self.blackout_prob:
            for _ in range(int(rng.integers(1, 4))):
                k0 = int(rng.integers(10, 400))
                self.dark.update(range(k0, k0 + int(rng.integers(2, 11))))
        self.per.reset(); self.exe.reset()
        self.s, self.prev_act, self.steps = s, np.zeros(ACT_DIM, dtype=np.float32), 0
        self.p = self.per.update(sim, s, 0 in self.dark)
        return observe(s, self.p["inv"])

    def step(self, act):
        act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
        prev = self.s
        s, fb = self.exe(self.sim, prev, decode(act), self.p)
        d_obs, over = obstacle_info(self.sim, s["pos"], s["goal"][2])
        em_net = bool(act[5] > 0)
        r = reward(prev, s, act, self.prev_act, d_obs, over, em_net, fb, self.em_bonus)
        self.s, self.prev_act = s, act
        self.steps += 1
        self.p = self.per.update(self.sim, s, self.steps in self.dark)
        goal = s["goal_dist"] < GOAL_RADIUS
        done = bool(s["crashed"]) or goal or self.steps >= self.max_steps
        info = dict(crashed=bool(s["crashed"]), reason=s["reason"], goal=goal, t=s["t"], H=s["H"],
                    E_motor=s["E_motor"], d_obs=d_obs, overfly=over, emergency=em_net, fallback=fb)
        return observe(s, self.p["inv"]), r, done, info

    def close(self):
        self.sim.close()
