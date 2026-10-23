"""
Полет мотопланера в окне MuJoCo. Управляет обученная сеть (ai_pilot.pth); если весов нет
или задан --baseline, работает базовый навигатор. Модуль включается автоматически при старте.
Уставки проходят через ai_common.Executor: крен ограничен и сглажен, предохранитель (fallback)
набирает высоту или разворачивает, если облет невозможен; препятствия видит стереокамера.

Сигнал оператору: при обнаружении препятствия в консоли звуковой сигнал и сообщение; след
траектории оранжевый, во время работы предохранителя или экстренного режима красный.

Консоль: X Y Z и Enter: новая цель (метры; x на восток, y на север, z вверх; в окне можно
перетащить красную сферу: двойной щелчок, затем правая кнопка с Ctrl);
  s  внезапное препятствие в 80 м по курсу;
  m  ручной режим: автоматическое уклонение выключено (курс на цель без обхода), повтор: включено;
  b  потеря изображения на 1 с (проверка прогноза по IMU);
  q  выход.
"""

import argparse
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_common import (D, GOAL_RADIUS, ROUTE_KINDS, AIPilot, Baseline, Executor,  # noqa: E402,F401  (Baseline нужен check_model.py)
                       Perception, route_scene)

def parse():
    ap = argparse.ArgumentParser(description="Полет мотопланера к цели в MuJoCo")
    ap.add_argument("--wind", nargs=2, type=float, default=[0.0, 0.0], metavar=("СКОРОСТЬ", "ОТКУДА"))
    ap.add_argument("--random-obstacles", action="store_true", help="случайные фазы и скорости движущихся препятствий")
    ap.add_argument("--speed", type=float, default=1.0, help="ускорение времени")
    ap.add_argument("--headless", action="store_true", help="без окна, только телеметрия")
    ap.add_argument("--duration", type=float, default=120.0, help="длительность без окна, с")
    ap.add_argument("--tz-scene", choices=ROUTE_KINDS, help="сцена методики ТЗ: 50 м, 15 м/с, препятствие на маршруте")
    ap.add_argument("--model", default="ai_pilot.pth", help="файл весов сети")
    ap.add_argument("--baseline", action="store_true", help="базовый навигатор вместо сети")
    ap.add_argument("--no-fallback", action="store_true", help="отключить предохранитель")
    ap.add_argument("--cpu", action="store_true", help="считать сеть на CPU")
    return ap.parse_args()

def build_policy(a):
    if not a.baseline:
        try:
            return AIPilot(a.model, cpu=a.cpu), f"сеть {a.model}"
        except Exception as exc:
            print(f"[демо] сеть не загружена ({type(exc).__name__}: {exc}), работает базовый навигатор")
    return Baseline(), "базовый навигатор"

def console(sim, lock, stop, ctl):
    for line in sys.stdin:
        p = line.replace(",", " ").split()
        if not p:
            continue
        cmd = p[0].lower()
        if cmd in ("q", "quit", "exit"):
            stop.set()
            break
        if cmd == "s":
            with lock:
                sim.spawn_obstacle(80.0)
            print("[консоль] внезапное препятствие в 80 м по курсу", flush=True)
            continue
        if cmd == "m":
            ctl["manual"] = not ctl["manual"]
            print(f"[консоль] {'РУЧНОЙ РЕЖИМ: автоматическое уклонение выключено' if ctl['manual'] else 'автоматическое уклонение включено'}",
                  flush=True)
            continue
        if cmd == "b":
            ctl["dark"] = 10
            print("[консоль] потеря изображения на 1 с", flush=True)
            continue
        try:
            xyz = [float(v) for v in p[:3]]
            with lock:
                sim.set_goal(xyz)
            print(f"[консоль] цель -> ({xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f})", flush=True)
        except (ValueError, IndexError):
            print("[консоль] нужно три числа: X Y Z", flush=True)

def trail_geoms(scn, pts):
    """След траектории: оранжевый обычный полет, красный уклонение предохранителем или экстренный режим."""
    import mujoco
    import numpy as np
    scn.ngeom = 0
    cols = (np.array([1.0, 0.65, 0.1, 1.0], np.float32), np.array([0.95, 0.1, 0.1, 1.0], np.float32))
    for (p0, _), (p1, hot) in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), cols[hot])
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0, np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

def main():
    a = parse()
    if a.headless and sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import mujoco
    import numpy as np
    import motoglider as mg

    policy, name = build_policy(a)
    sim, per, rng = mg.GliderSim(depth=True), Perception(), np.random.default_rng()
    opts = dict(wind=tuple(a.wind), randomize_obstacles=a.random_obstacles)

    def new_flight():
        if a.tz_scene:
            return route_scene(sim, rng, a.tz_scene, wind=(a.wind[0], a.wind[1]))[0]
        return sim.reset(**opts)

    s = new_flight()
    exe, manual, ctl = Executor(not a.no_fallback), Executor(False), dict(manual=False, dark=0)
    hand = Baseline(avoid=False)
    p = per.update(sim, s)
    lock, stop = threading.Lock(), threading.Event()
    threading.Thread(target=console, args=(sim, lock, stop, ctl), daemon=True).start()
    print(f"[демо] пилот: {name}; цель: {s['goal'].round(1)}; ввод: X Y Z, s, m, b, q")
    trail, nxt, alarm = deque(maxlen=900), dict(p=0.0, tr=0.0), [False]

    def advance():
        nonlocal s, p
        with lock:
            if ctl["manual"]:
                s, fb = manual(sim, s, hand(sim, s), p)
            else:
                s, fb = exe(sim, s, policy(sim, s, p), p, getattr(policy, "unsure", False))
            ctl["dark"] = max(0, ctl["dark"] - 1)
            p = per.update(sim, s, ctl["dark"] > 0)
        if p["threat"] and not alarm[0]:
            print(f"\a[ВНИМАНИЕ] препятствие по курсу: {p['front']:.0f} м", flush=True)
        alarm[0] = p["threat"]
        if s["t"] >= nxt["tr"]:
            trail.append((s["pos"].copy(), int(fb or s["emergency"])))
            nxt["tr"] = s["t"] + 0.5
        if s["t"] >= nxt["p"]:
            tag = (" ЭКСТР" if s["emergency"] and not fb else "") + (" FALLBACK" if fb else "") + \
                  (" РУЧНОЙ" if ctl["manual"] else "") + (" НЕТ КАДРА" if p["lost"] else "")
            print(f"t={s['t']:6.1f} с  x={s['pos'][0]:7.0f} y={s['pos'][1]:6.0f} H={s['H']:5.0f} м "
                  f"(цель {s['goal'][2]:4.0f})  V={s['V']:4.1f} м/с  Vy={s['vel'][2]:+5.2f}  крен {s['gamma'] / D:+5.1f}  "
                  f"мотор {'ВКЛ ' if s['motor_on'] else 'выкл'}  заряд {s['battery'] * 100:3.0f} %  "
                  f"до цели {s['goal_dist']:5.0f} м{tag}", flush=True)
            nxt["p"] = s["t"] + 1.0
        if s["crashed"] or s["goal_dist"] < GOAL_RADIUS:
            print(f"[полет] {'авария: ' + s['reason'] if s['crashed'] else 'цель достигнута'} на {s['t']:.0f} с", flush=True)
            goal = s["goal"]
            with lock:
                s = new_flight()
                if not a.tz_scene:
                    sim.set_goal(goal)
                    s = sim.state()
                per.reset()
                p = per.update(sim, s)
            policy.reset(); exe.reset(); manual.reset(); hand.reset(); trail.clear()
            nxt.update(p=0.0, tr=0.0)

    try:
        if a.headless:
            scn = mujoco.MjvScene(sim.model, maxgeom=2000)
            while sim.t < a.duration and not stop.is_set():
                advance()
                trail_geoms(scn, list(trail))
            print(f"[проверка] шагов {int(round(sim.t / 0.1))}, линий следа {scn.ngeom}")
            return 0
        from mujoco import viewer
        with viewer.launch_passive(sim.model, sim.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = sim.bid
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 14.0, -15.0, 180.0
            while v.is_running() and not stop.is_set():
                t0 = time.time()
                with v.lock():
                    advance()
                    trail_geoms(v.user_scn, list(trail))
                v.sync()
                time.sleep(max(0.0, 0.1 / max(a.speed, 0.05) - (time.time() - t0)))
        return 0
    finally:
        sim.close()

if __name__ == "__main__":
    sys.exit(main())
