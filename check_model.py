"""
Проверка готовой модели мотопланера одной командой:  python check_model.py
Все проверки обязательные. Итог на один экран строится из фактических прогонов,
код возврата 0 только при полностью зеленом итоге. Снимки и журнал: папка check_output.
"""

import math
import os
import struct
import sys
import time
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "check_output")
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, HERE)
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "check_log.txt"), "w", encoding="utf-8")
FILES = ("motoglider.xml", "motoglider.py", "check_model.py", "fly_demo.py", "README.txt")
CHECKS = []
IMPORT_ERR = None
try:
    import numpy as np
    import mujoco
    import motoglider as mg
except Exception as exc:  # pragma: no cover
    IMPORT_ERR = f"{type(exc).__name__}: {exc}"
    traceback.print_exc(file=LOG)

D = math.pi / 180.0

def check(title):
    def deco(fn):
        CHECKS.append((title, fn))
        return fn
    return deco

def log(*a):
    print(*a, file=LOG)
    LOG.flush()

def save_png(path, img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def ch(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + ch(b"IDAT", zlib.compress(raw, 6)) + ch(b"IEND", b""))

def fly(sim, n, gamma=0.0, V=12.0, beta=0.0, thr=0.0):
    rec = []
    for _ in range(n):
        s = sim.step(gamma * D, V, beta * D, thr)
        rec.append(s)
        if s["crashed"]:
            break
    return rec

@check("Окружение и состав пакета")
def c_env():
    if IMPORT_ERR:
        return False, f"не импортируется: {IMPORT_ERR}; установка: pip install mujoco numpy"
    miss = [f for f in FILES if not os.path.isfile(os.path.join(HERE, f))]
    mv = tuple(int(x) for x in mujoco.__version__.split(".")[:2])
    return (not miss and mv >= (3, 1)), (f"Python {sys.version.split()[0]}, MuJoCo {mujoco.__version__}, numpy {np.__version__}"
                                         + (f"; нет файлов: {', '.join(miss)}" if miss else ""))

@check("Модель MJCF: массы, рули, камеры, препятствия")
def c_model():
    sim = mg.GliderSim()
    m, d = sim.model, sim.data
    cl, cr = m.camera("cam_left").id, m.camera("cam_right").id
    base = float(np.linalg.norm(d.cam_xpos[cl] - d.cam_xpos[cr]))
    fovy = float(m.cam_fovy[cl])
    ok = (abs(sim._mass() - 4.0) < 1e-3 and m.nu == 4 and abs(fovy - 46.397) < 0.05 and abs(base - sim.B) < 1e-3
          and len(sim.movers) == 4 and len(sim.static) == 6)
    return ok, (f"масса {sim._mass():.3f} кг, приводов {m.nu} (элероны, РВ, РН), камеры 1280x800 поле 68,9x{fovy:.1f} град, "
                f"база {base:.3f} м, препятствий: статичных {len(sim.static)}, движущихся {len(sim.movers)}")

@check("Поляра и контрольные характеристики")
def c_polar():
    sim = mg.GliderSim()
    p = mg.polar(sim)
    K12 = mg.glide_K(sim, 12.0)[0]
    tgt = [("Kmax", p["K_max"], 17.0, 0.3), ("V(Kmax)", p["V_Kmax"], 9.8, 0.2), ("Vсв", p["V_stall"], 8.04, 0.1),
           ("Vy min", p["sink_min"], 0.53, 0.02), ("K(12)", K12, 15.64, 0.2)]
    bad = [n for n, v, t, tol in tgt if abs(v - t) > tol]
    return not bad, (f"Kmax {p['K_max']:.2f} при {p['V_Kmax']:.2f} м/с, мин. снижение {p['sink_min']:.3f} м/с при "
                     f"{p['V_sink']:.2f} м/с, сваливание {p['V_stall']:.2f} м/с, K(12 м/с) {K12:.2f}" + (f"; вне допуска: {bad}" if bad else ""))

@check("Знаки сил и моментов по ГОСТ 20058-80")
def c_signs():
    sim = mg.GliderSim()
    a = 4 * D
    Vg = (12 * math.cos(a), -12 * math.sin(a), 0.0)
    z = (0.0, 0.0, 0.0)
    F, M0, _ = sim.forces(Vg, z, z, 1.225)
    errs = []
    vhat = np.array([math.cos(a), -math.sin(a), 0.0])
    lhat = np.array([math.sin(a), math.cos(a), 0.0])
    if not (F @ vhat < 0 and F @ lhat > 0):
        errs.append("подъемная сила или сопротивление")
    if not sim.forces(Vg, z, (0.1, 0, 0), 1.225)[1][0] < M0[0]:
        errs.append("элероны: δэ > 0 должен давать mx < 0")
    if not sim.forces(Vg, z, (0, 0.1, 0), 1.225)[1][2] < M0[2]:
        errs.append("руль высоты: δв > 0 должен давать mz < 0")
    if not sim.forces(Vg, z, (0, 0, 0.1), 1.225)[1][1] < M0[1]:
        errs.append("руль направления: δн > 0 должен давать my < 0")
    Vb = (12 * math.cos(a), -12 * math.sin(a), 12 * math.sin(3 * D))
    Fb, Mb, _ = sim.forces(Vb, z, z, 1.225)
    if not (Mb[1] < M0[1] and Mb[0] < M0[0] and Fb[2] < F[2]):
        errs.append("скольжение: флюгерная и поперечная устойчивость, боковая сила")
    for i, w in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        if not sim.forces(Vg, w, z, 1.225)[1][i] < M0[i]:
            errs.append(f"демпфирование по оси {'XYZ'[i]}")
    if abs(np.linalg.det(mg.GOST) - 1) > 1e-12:
        errs.append("переход к осям ГОСТ")
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = (rng.uniform(-1, 1), rng.uniform(-1.2, 1.2), rng.uniform(-3, 3))
        if max(abs(x - y) for x, y in zip(mg.attitude(mg.matrix(*e)), e)) > 1e-9:
            errs.append("углы ориентации")
            break
    return not errs, ("подъемная сила вверх, mx^δэ, my^δн, mz^δв отрицательны, флюгер, демпфирование, оси ГОСТ"
                      if not errs else "; ".join(errs))

@check("Планирование: качество в полете равно поляре")
def c_glide():
    sim = mg.GliderSim()
    sim.reset(start=(0, 0, 400), airspeed=12.0)
    rec = fly(sim, 700, V=12.0)
    s = [r for r in rec if r["t"] > 30]
    V = np.mean([r["V"] for r in s])
    vh = np.mean([math.hypot(*r["vel"][:2]) for r in s])
    vz = np.mean([r["vel"][2] for r in s])
    Kp = mg.glide_K(sim, V, rho=sim._rho(np.mean([r["H"] for r in s])))[0]
    ok = not sim.crashed and abs(vh / -vz / Kp - 1) < 0.03 and np.std([r["V"] for r in s]) < 0.3
    return ok, f"K = {vh / -vz:.2f} при {V:.2f} м/с (поляра {Kp:.2f}), снижение {-vz:.2f} м/с"

@check("Мотор: набор высоты и расход энергии")
def c_motor():
    sim = mg.GliderSim()
    sim.reset(start=(0, 0, 200), airspeed=12.0)
    fly(sim, 50, V=12.0)
    t0, E0 = sim.t, sim.E_motor
    rec = fly(sim, 300, V=12.0, thr=1.0)
    s = [r for r in rec if r["t"] > t0 + 10]
    tt = np.array([r["t"] for r in s])
    climb = float(np.polyfit(tt, [r["H"] for r in s], 1)[0])
    V = float(np.mean([r["V"] for r in s]))
    P = float(np.mean([r["thrust"] for r in s]))
    rho = sim._rho(np.mean([r["H"] for r in s]))
    K = mg.glide_K(sim, V, rho=rho)[0]
    W = sim._mass() * mg.G0
    theory = (P - W / K) * V / W
    Eexp = sum(sim.thrust(1.0 if r["motor_on"] else 0.0, r["V"], sim._rho(r["H"]))[1] * 0.1 for r in rec if r["t"] > t0 + 2)
    Emod = sim.E_motor - E0
    ok = climb >= 2.5 and abs(climb / theory - 1) < 0.15 and abs(Emod / max(Eexp, 1) - 1) < 0.08
    return ok, (f"набор {climb:.2f} м/с (расчет {theory:.2f}) на {V:.1f} м/с, тяга {P:.1f} Н, "
                f"расход {Emod / 1000:.1f} кДж за 30 с (по формуле {Eexp / 1000:.1f})")

@check("Каналы ПИД: крен, скорость, скольжение, газ")
def c_pid():
    sim = mg.GliderSim()
    sim.reset(start=(0, 0, 300), airspeed=12.0)
    fly(sim, 30)
    rec = fly(sim, 60, gamma=30)
    g = np.array([r["gamma"] for r in rec]) / D
    b = np.array([r["beta"] for r in rec]) / D
    t90 = next((i * 0.1 for i, x in enumerate(g) if x >= 27), 99)
    sim.reset(start=(0, 0, 400), airspeed=12.0)
    fly(sim, 100)
    v = np.array([r["V"] for r in fly(sim, 250, V=15.0)])
    tv = next((i * 0.1 for i, x in enumerate(v) if x >= 14.7), 99)
    over, err = v.max() - 15.0, abs(v[200:].mean() - 15.0)
    sim.reset(start=(0, 0, 300), airspeed=12.0)
    fly(sim, 30)
    bs = np.mean([r["beta"] for r in fly(sim, 80, beta=5.0)[-20:]]) / D
    sim.reset(start=(0, 0, 300), airspeed=12.0)
    t_on = next((r["t"] for r in fly(sim, 10, thr=1.0) if r["thrust"] > 0.5), 99)
    ok = (t90 <= 2.0 and g.max() <= 34.5 and abs(b).max() <= 2.0 and tv <= 8.0 and over <= 1.0 and err <= 0.3
          and abs(bs - 5) <= 1.5 and t_on <= 0.7)
    return ok, (f"крен 30 град за {t90:.1f} с (максимум {g.max():.1f}), скольжение в развороте до {abs(b).max():.1f} град, "
                f"скорость 12 -> 15 м/с за {tv:.1f} с (заброс {max(over, 0):.2f}, ошибка {err:.2f} м/с), "
                f"скольжение 5 град держится как {bs:.1f}, тяга через {t_on:.1f} с")

@check("Движущиеся препятствия: законы движения")
def c_movers():
    sim = mg.GliderSim()
    sim.reset(start=(0, 0, 300), airspeed=12.0, randomize_obstacles=True)
    errs, speeds = 0.0, []
    for _ in range(6):
        fly(sim, 10, V=12.0)
        for mv in sim.movers:
            p, v = sim.mover_state(mv, sim.t)
            xp = sim.data.mocap_pos[mv["mocap"]]
            errs = max(errs, float(np.linalg.norm(xp - p)))
            p2, _ = sim.mover_state(mv, sim.t + 1e-3)
            errs = max(errs, float(np.linalg.norm((p2 - p) / 1e-3 - v)) * 0.01)
            speeds.append((mv["name"], float(np.linalg.norm(v))))
    vmax = {}
    for n_, s_ in speeds:
        vmax[n_] = max(vmax.get(n_, 0.0), s_)
    txt = ", ".join(f"{k.replace('mov_', '')} до {v:.1f} м/с" for k, v in vmax.items())
    return errs < 0.01, f"положения и скорости совпадают с формулами (ошибка {errs:.4f} м); {txt}"

@check("Столкновения: башня и движущийся аэростат")
def c_crash():
    sim = mg.GliderSim()
    sim.reset(start=(580, -70, 60), airspeed=12.0)
    r1 = fly(sim, 200, V=12.0)[-1]
    sim.reset(start=(0, 0, 300), airspeed=12.0)
    mv = next(m_ for m_ in sim.movers if m_["name"] == "mov_balloon_1")
    p10, _ = sim.mover_state(mv, 10.0)
    sim.reset(start=(p10[0] - 120, p10[1], p10[2] + 2), airspeed=12.0)
    r2 = fly(sim, 200, V=12.0)[-1]
    ok = r1["crashed"] and r1["reason"] == "препятствие" and r2["crashed"] and "mov_balloon_1" in r2["reason"]
    return ok, f"башня: {r1['reason']} на {r1['t']:.1f} с; аэростат: {r2['reason']} на {r2['t']:.1f} с"

@check("Камеры: глубина, ошибки стереопары, снимки")
def c_cam():
    sim = mg.GliderSim(depth=True)
    rends = []
    try:
        sim.reset(start=(594, -70, 60), airspeed=12.0)
        m, d = sim.model, sim.data
        cid = m.camera("cam_left").id
        r_odd = mujoco.Renderer(m, 41, 65)
        rends.append(r_odd)
        z = sim.depth_true(renderer=r_odd)
        axis = -d.cam_xmat[cid].reshape(3, 3)[:, 2]
        gid = np.zeros(1, dtype=np.int32)
        ray = mujoco.mj_ray(m, d, d.cam_xpos[cid].copy(), axis.copy(), sim.ray_group, 1, sim.bid, gid)
        Zs = sim.stereo_errors(np.full((200, 300), 100.0), 1.0)
        sig = float(np.std(Zs[np.isfinite(Zs)]))
        r_full = mujoco.Renderer(m, 800, 1280)
        rends.append(r_full)
        imgs = []
        for cam in ("cam_left", "cam_right"):
            r_full.update_scene(d, camera=cam, scene_option=sim.scene_opt)
            imgs.append(r_full.render().copy())
        dep = sim.depth()
        inv = np.where(np.isfinite(dep), np.clip(20.0 / np.maximum(dep, 1.0), 0, 1), 0)
        big = np.kron((inv * 255).astype(np.uint8), np.ones((10, 10), dtype=np.uint8))
        save_png(os.path.join(OUT, "stereo_views.png"), np.concatenate([imgs[0][::2, ::2], imgs[1][::2, ::2], np.stack([big] * 3, -1)], 1))
        sim.reset(airspeed=12.0)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.qpos[0:3]
        cam.distance, cam.azimuth, cam.elevation = 5.5, 140.0, -14.0
        r_full.update_scene(d, camera=cam)
        save_png(os.path.join(OUT, "glider.png"), r_full.render())
        cam.lookat[:] = [850.0, 0.0, 60.0]
        cam.distance, cam.azimuth, cam.elevation = 1250.0, 235.0, -14.0
        r_full.update_scene(d, camera=cam)
        save_png(os.path.join(OUT, "scene.png"), r_full.render())
        ok = abs(z[20, 32] / ray - 1) < 0.005 and abs(sig / sim.sigma_D(100.0) - 1) < 0.2 and imgs[0].std() > 5 and np.isfinite(dep).any()
        return ok, (f"глубина до башни {z[20, 32]:.1f} м (луч {ray:.1f}), СКО на 100 м {sig:.2f} м "
                    f"(формула {sim.sigma_D(100.0):.2f}), снимки сохранены")
    finally:
        for r in rends:
            r.close()
        sim.close()

@check("Устойчивость: 30 случайных эпизодов с ветром и случайными уставками")
def c_stress():
    sim = mg.GliderSim(seed=11)
    rng = np.random.default_rng(5)
    steps, nan, why = 0, 0, {}
    for _ in range(30):
        sim.reset(start=(0, 0, rng.uniform(80, 250)), heading_deg=rng.uniform(0, 360), airspeed=rng.uniform(10, 16),
                  wind=(rng.uniform(0, 7), rng.uniform(0, 360)), randomize_obstacles=True)
        for _ in range(150):
            s = sim.step(rng.uniform(-45, 45) * D, rng.uniform(9.5, 20), rng.uniform(-10, 10) * D, rng.uniform(0, 1))
            steps += 1
            nan += int(not np.all(np.isfinite(s["pos"])))
            if s["crashed"]:
                why[s["reason"].split(" ")[0]] = why.get(s["reason"].split(" ")[0], 0) + 1
                break
    bad = why.get("неустойчивость", 0)
    return bad == 0 and nan == 0, (f"{steps} шагов: неустойчивость {bad}, нечисла {nan}; исходы: "
                                   + (", ".join(f"{k} {v}" for k, v in sorted(why.items())) or "без аварий"))

def avoid_threshold(sim, mode):
    """Наименьшая дистанция появления шара 6 м по курсу, с которой маневр уходит и планер цел 15 с.
    Экстренный режим: руль высоты вниз 15 град на 0,5 с; обычный: уставка 20 м/с на 2 с; крен 45 град 3 с."""
    dur = 5 if mode == "em" else 20
    for dist in range(10, 46):
        s = sim.reset(start=(0, 0, 120), airspeed=12.0)
        fly(sim, 20)
        sim.spawn_obstacle(float(dist))
        for k in range(150):
            g = 45 * D if k < 30 else 0.0
            if mode == "em":
                s = sim.step(g, 12.0, 0.0, 0.0, elevator=15 * D if k < dur else None)
            else:
                s = sim.step(g, 20.0 if k < dur else 12.0, 0.0, 0.0)
            if s["crashed"]:
                break
        if not s["crashed"]:
            return dist
    return 99

@check("Экстренный режим: прямой руль высоты при внезапном препятствии")
def c_emerg():
    sim = mg.GliderSim()
    sim.reset(start=(0, 0, 300), airspeed=12.0)
    fly(sim, 20)
    for _ in range(2):
        sim.step(0.0, 12.0, 0.0, 0.0, elevator=10 * D)
    follow = abs(sim.state()["deflections"][1] / D - 10.0)
    for _ in range(3):
        sim.step(0.0, 12.0, 0.0, 0.0, elevator=10 * D)
    dv = [sim.state()["deflections"][1] / D]
    rec = fly(sim, 150)
    dv += [r["deflections"][1] / D for r in rec]
    jump = float(np.max(np.abs(np.diff(dv[:20]))))
    t_back = next((r["t"] - rec[0]["t"] for r in rec if abs(r["V"] - 12.0) < 1.0 and r["t"] - rec[0]["t"] > 2), 99)
    intact = not rec[-1]["crashed"]
    sim.reset(start=(0, 0, 300), airspeed=12.0)
    fly(sim, 20)
    for _ in range(20):
        sim.step(0.0, 12.0, 0.0, 0.0, elevator=10 * D)
    dive = fly(sim, 150)[-1]
    sim.reset(start=(0, 0, 120), airspeed=12.0)
    fly(sim, 20)
    sim.spawn_obstacle(60.0)
    hit = fly(sim, 80)[-1]
    d_em, d_std = avoid_threshold(sim, "em"), avoid_threshold(sim, "std")
    ok = (follow <= 0.5 and jump <= 3.0 and t_back <= 15.0 and intact and dive["crashed"] and "перегрузка" in dive["reason"]
          and hit["crashed"] and "внезапное" in hit["reason"] and d_em < d_std)
    return ok, (f"руль высоты отрабатывает команду с ошибкой {follow:.2f} град; после 0,5 с руля вниз 10 град возврат к скорости: "
                f"шаг руля до {jump:.1f} град за 0,1 с, скорость в пределах 1 м/с через {t_back:.1f} с, планер цел; "
                f"2 с руля вниз 10 град: {dive['reason'] or 'без разрушения'}; шар 6 м без реакции сбивает планер; "
                f"уход с креном 45 град: экстренный режим с {d_em} м, обычный режим с {d_std} м")

@check("Случайные сцены: препятствия в коридоре маршрута, столкновения")
def c_random():
    sim = mg.GliderSim(seed=21)
    rng = np.random.default_rng(3)
    bad = 0
    for _ in range(20):
        start, goal, hdg = sim.randomize_scene(rng)
        u = (goal[:2] - start[:2]) / np.linalg.norm(goal[:2] - start[:2])
        for k in range(len(sim.static)):
            p = sim.scene_pos[k][:2] - start[:2]
            along, across = p @ u, abs(p[0] * u[1] - p[1] * u[0])
            if along < 100 or across > 200 + 1e-6:
                bad += 1
    k = next(i for i, g in enumerate(sim.static) if sim.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
    gid = sim.static[k]
    p, r, hh = sim.scene_pos[k].copy(), sim.model.geom_size[gid][0], sim.model.geom_size[gid][1]
    res = []
    for dy in (r - 0.5, r + 3.5):
        sim.reset(start=(p[0] - r - 70, p[1] + dy, hh), heading_deg=90.0, airspeed=12.0)
        res.append(fly(sim, 160)[-1])
    sim.default_scene()
    m2 = mg.GliderSim()
    same = np.allclose(sim.scene_pos, m2.scene_pos) and np.allclose(sim.model.geom_size, m2.model.geom_size)
    ok = bad == 0 and res[0]["crashed"] and res[0]["reason"] == "препятствие" and not res[1]["crashed"] and same
    return ok, (f"20 случайных сцен: препятствий вне коридора {bad}; башня случайного размера радиусом {r:.1f} м: "
                f"пролет в {r - 0.5:.1f} м от оси {'засчитан как удар' if res[0]['crashed'] else 'НЕ засчитан'}, "
                f"в {r + 3.5:.1f} м {'без удара' if not res[1]['crashed'] else 'С УДАРОМ'}; возврат к сцене по умолчанию {'точный' if same else 'НЕТОЧНЫЙ'}")

@check("Базовый навигатор долетает до цели")
def c_nav():
    import fly_demo
    sim = mg.GliderSim()
    s = sim.reset(airspeed=12.0)
    nav = fly_demo.Baseline()
    for _ in range(3000):
        s = sim.step(*nav(sim, s))
        if s["crashed"] or s["goal_dist"] < 30:
            break
    ok = s["goal_dist"] < 30 and not s["crashed"] and s["t"] < 300
    return ok, (f"сцена по умолчанию: {'цель достигнута' if s['goal_dist'] < 30 else 'цель НЕ достигнута'} за {s['t']:.0f} с, "
                f"{'авария: ' + s['reason'] if s['crashed'] else 'без столкновений'}, мотор {s['E_motor'] / 1000:.1f} кДж")

@check("Скорость счета")
def c_speed():
    sim = mg.GliderSim()
    sim.reset(start=(0, 0, 400), airspeed=12.0)
    t0 = time.time()
    fly(sim, 300, gamma=10, V=12.0)
    sps = 300 / (time.time() - t0)
    return sps >= 50, f"{sps:.0f} шагов по 0,1 с в секунду на одно ядро ({sps * 10:.0f} шагов физики)"

@check("Оформление файлов")
def c_style():
    bad = [f for f in FILES if os.path.isfile(os.path.join(HERE, f)) and any(
        ch in open(os.path.join(HERE, f), encoding="utf-8").read() for ch in ("\u0451", "\u0401", "\u2014", "\u2013"))]
    return not bad, "нет буквы йо и длинных тире" if not bad else "нарушения: " + ", ".join(bad)

def main():
    t0 = time.time()
    res = []
    for i, (title, fn) in enumerate(CHECKS, 1):
        if IMPORT_ERR and i > 1:
            ok, msg = False, "не выполнялась: нет зависимостей"
        else:
            try:
                ok, msg = fn()
            except Exception as exc:
                ok, msg = False, f"исключение {type(exc).__name__}: {exc}"
                traceback.print_exc(file=LOG)
        res.append((i, title, bool(ok), msg))
        log(f"[{i}] {title}: {'OK' if ok else 'FAIL'} {msg}")
        print(f"  {i:2d}/{len(CHECKS)} {title} ...", "ok" if ok else "ОШИБКА", flush=True)
    n_ok = sum(r[2] for r in res)
    line = "=" * 100
    print(f"\n{line}\nПРОВЕРКА МОДЕЛИ МОТОПЛАНЕРА (motoglider.xml)\n{line}")
    for i, title, ok, msg in res:
        print(f"{'✓' if ok else '✗'} {i:2d}. {title}\n       {msg}")
    print("-" * 100)
    if n_ok == len(res):
        print(f"ИТОГ: ✓ пройдены все {len(res)} проверок, код возврата 0  (время {time.time() - t0:.0f} с)")
    else:
        print(f"ИТОГ: ✗ не пройдено {len(res) - n_ok} из {len(res)} (номера: {', '.join(str(r[0]) for r in res if not r[2])}), "
              f"код возврата 1  (журнал: check_output/check_log.txt)")
    print(line)
    LOG.close()
    return 0 if n_ok == len(res) else 1

if __name__ == "__main__":
    sys.exit(main())
