"""
Испытания пилота и протокол по метрикам ТЗ.

Метрики: столкновения, зазор от законцовки крыла до препятствия (норма >= 3 м), фактический крен
(норма <= 30 град), время реакции от обнаружения до начала маневра (норма <= 300 мс, ни один
замер > 500 мс), время обработки кадра (восприятие + сеть, норма >= 20 кадров/с), пролет над
препятствием, отклонение от высоты цели, доля экстренного режима и предохранителя.
"""
import argparse
import csv
import math
import os
import time
import warnings
import gc

import numpy as np
import motoglider as mg
from ai_common import (D, GOAL_RADIUS, ROUTE_KINDS, AIPilot, Baseline, Executor, Perception, obstacle_info,
                       route_scene)

warnings.simplefilter("ignore", RuntimeWarning)       # nanmean по пустым выборкам
SCENARIOS = ("tz20", "default", "wind8", "sudden", "blackout", "random")
FIELDS = ("сценарий", "препятствие", "авария", "причина", "цель", "t_с", "E_кДж", "зазор_м", "крен_град",
          "реакция_ср_мс", "реакция_макс_мс", "кадр_ср_мс", "кадр_макс_мс", "кадров_ниже_15Гц",
          "над_препятствием_%", "dH_м", "экстр_%", "fallback_%")
TRAJ = ("t", "x", "y", "z", "V", "крен_град", "до_цели", "до_препятствия", "угроза", "fallback", "экстр", "потеря_кадра")


def start(sim, sc, rng, kind):
    if sc in ("tz20", "blackout"):
        return route_scene(sim, rng, kind, H=50.0, L=600.0, V=15.0, wind=(rng.uniform(0, 8), rng.uniform(0, 360)),
                           bearing=rng.uniform(0, 360))
    if sc == "random":
        st, goal, hdg = sim.randomize_scene(rng)
        return sim.reset(start=st, heading_deg=hdg, goal=goal, randomize_obstacles=True,
                         wind=(rng.uniform(0, 8), rng.uniform(0, 360))), "случайная сцена"
    sim.default_scene()
    s = sim.reset(wind=(8, 270) if sc == "wind8" else (0, 0))
    if sc == "sudden":
        sim.spawn_obstacle(25.0)
    return s, "сцена по умолчанию"

def episode(policy, exe, per, sim, sc, rng, kind=None, traj=None, max_steps=2500):                                # очистка мусора до эпизода (без disable)
    s, kind = start(sim, sc, rng, kind)
    policy.reset(); exe.reset(); per.reset()
    p, dark_from = per.update(sim, s), None
    d_min, bank, over, em, fb, dh, proc = 1e3, 0.0, 0, 0, 0, [], []
    import gc
    gc.collect()
    for k in range(max_steps):
        t0 = time.perf_counter()
        sp = policy(sim, s, p)
        p["t_proc"] += time.perf_counter() - t0
        if k >= 20:  # пропуск JIT-прогрева 2 с
            proc.append(p["t_proc"] * 1000)
        if k > 0 and k % 500 == 0:  # GC в известный момент
            gc.collect()
        s, f = exe(sim, s, sp, p, getattr(policy, "unsure", False))
        d, o = obstacle_info(sim, s["pos"], s["goal"][2])
        e = len(sp) > 4 and sp[4] is not None
        d_min, bank, over, em, fb = min(d_min, d), max(bank, abs(s["gamma"])), over + o, em + e, fb + f
        if s["t"] > 20:
            dh.append(abs(s["H"] - s["goal"][2]))
        if traj is not None:
            traj.append([round(x, 3) for x in (s["t"], *s["pos"], s["V"], s["gamma"] / D, s["goal_dist"], d)]
                        + [int(p["threat"]), int(f), int(e), int(p["lost"])])
        if s["crashed"] or s["goal_dist"] < GOAL_RADIUS:
            break
        if sc == "blackout" and dark_from is None and p["threat"]:
            dark_from = k + 1
        p = per.update(sim, s, dark_from is not None and dark_from <= k + 1 < dark_from + 10)
    n, rc = k + 1, exe.reactions
    return dict(
        zip(FIELDS, (sc, kind, int(s["crashed"]), s["reason"], int(s["goal_dist"] < GOAL_RADIUS and not s["crashed"]),
                     s["t"], s["E_motor"] / 1000, d_min - sim.l / 2, bank / D,
                     float(np.mean(rc)) if rc else math.nan, max(rc) if rc else math.nan,
                     float(np.mean(proc)) if proc else math.nan, max(proc) if proc else math.nan,
                     sum(x > 1000 / 15 for x in proc) if proc else 0,
                     100 * over / n, float(np.mean(dh)) if dh else math.nan, 100 * em / n, 100 * fb / n)))
def verdict(rows):
    """Вердикт по таблице параметров ТЗ для прогонов сценария tz20."""
    n = len(rows)
    col = lambda f: np.array([r[f] for r in rows], dtype=float)                       # noqa: E731
    crash, clr, bank = col("авария"), col("зазор_м"), col("крен_град")
    rmean, rmax = col("реакция_ср_мс"), col("реакция_макс_мс")
    react = np.nanmean(rmean) if np.isfinite(rmean).any() else math.nan
    ok = lambda c: "✓" if c else "✗"                                                    # noqa: E731
    succ = 100 * (1 - crash.mean())
    out = [(ok(succ >= 95), f"1. доля уклонений без столкновения {succ:.0f} % (норма >= 95 %; 90-94 % требует анализа)"),
           (ok(react <= 350 and np.nanmax(rmax) <= 500), f"2. время реакции: среднее {react:.0f} мс, максимум {np.nanmax(rmax):.0f} мс "
                                                       f"(норма <= 300, допуск среднего 350, ни один замер > 500)"),
           (ok((clr >= 2.5).all() and (clr < 3).sum() <= 2), f"3. минимальный зазор {clr.min():.1f} м, прогонов с зазором 2,5-3 м: "
                                                          f"{int(((clr >= 2.5) & (clr < 3)).sum())} (норма >= 3 м, допуск 2,5 м не более чем в 2)"),
           (ok(col("кадр_ср_мс").mean() <= 50 and col("кадров_ниже_15Гц").max() <= 60),
            f"4. обработка кадра: в среднем {col('кадр_ср_мс').mean():.1f} мс ({1000 / max(col('кадр_ср_мс').mean(), 1e-3):.0f} кадров/с), "
            f"кадров медленнее 15 Гц: {int(col('кадров_ниже_15Гц').max())} из ~{int(col('t_с').sum() * 10)} "
            f"(норма >= 20 кадров/с, до 15 Гц не более 3 раз за прогон)"),
           (ok((bank <= 35).all() and (bank > 30).sum() <= 0.1 * n), f"5. максимальный крен {bank.max():.1f} град, прогонов с креном 30-35 град: "
                                                                    f"{int((bank > 30).sum())} из {n} (норма <= 30, до 35 не более чем в 10 %)")]
    print("\nВердикт по ТЗ (сценарий tz20):")
    for mark, text in out:
        print(f"  {mark} {text}")
    print(f"  итог: {'все параметры выполнены' if all(m == '✓' for m, _ in out) else 'есть невыполненные параметры'}")

def evaluate(a):
    gc.freeze()  # все существующие объекты в permanent generation, GC их не обходит
    policy = Baseline() if a.baseline else AIPilot(a.model, cpu=a.cpu)
    sim, per = mg.GliderSim(seed=a.seed, depth=True), Perception()
    exe, rng = Executor(not a.no_fallback), np.random.default_rng(a.seed)
    if a.log:
        os.makedirs(a.log, exist_ok=True)
    print(f"Пилот: {'базовый навигатор' if a.baseline else a.model}; предохранитель {'выкл' if a.no_fallback else 'вкл'}")
    head = ("сценарий", "без авар", "цель", "зазор, м", "крен", "реакц ср", "реакц макс", "кадр, мс", "над преп %", "dH, м", "fb, %")
    print("".join(f"{h:>11}" for h in head))
    rows_all = []
    for sc in a.only or SCENARIOS:
        if sc == "tz20":
            kinds = list(rng.permutation(np.repeat(ROUTE_KINDS, 4)))
        else:
            kinds = [None] * (a.n_random if sc == "random" else a.n_simple)
        rows = []
        for i, kind in enumerate(kinds):
            traj = [] if a.log else None
            rows.append(episode(policy, exe, per, sim, sc, rng, kind, traj))
            if a.log:
                with open(os.path.join(a.log, f"traj_{sc}_{i:02d}.csv"), "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerows([TRAJ, *traj])
        rows_all += rows
        c = lambda f: np.array([r[f] for r in rows], dtype=float)                      # noqa: E731
        n = len(rows)
        cells = (sc, f"{n - int(c('авария').sum())}/{n}", f"{int(c('цель').sum())}/{n}", f"{c('зазор_м').min():.1f}",
                 f"{c('крен_град').max():.1f}", f"{np.nanmean(c('реакция_ср_мс')):.0f}", f"{np.nanmax(c('реакция_макс_мс')):.0f}",
                 f"{c('кадр_макс_мс').max():.1f}", f"{c('над_препятствием_%').mean():.1f}", f"{np.nanmean(c('dH_м')):.1f}",
                 f"{c('fallback_%').mean():.1f}")
        print("".join(f"{x:>11}" for x in cells), flush=True)
    for r in rows_all:
        if r["авария"]:
            print(f"  авария: {r['сценарий']}, {r['препятствие']}: {r['причина']} на {r['t_с']:.0f} с")
    if a.log:
        with open(os.path.join(a.log, "protocol.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, FIELDS)
            w.writeheader()
            w.writerows(rows_all)
        print(f"Протокол: {os.path.join(a.log, 'protocol.csv')}, траектории: traj_*.csv")
    tz = [r for r in rows_all if r["сценарий"] == "tz20"]
    if tz:
        verdict(tz)
    sim.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ai_pilot.pth")
    ap.add_argument("--baseline", action="store_true", help="испытать базовый навигатор")
    ap.add_argument("--no-fallback", action="store_true", help="без предохранителя")
    ap.add_argument("--only", nargs="+", choices=SCENARIOS, help="только эти сценарии")
    ap.add_argument("--log", help="папка для протокола и траекторий")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n-simple", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    evaluate(ap.parse_args())
