"""
Общая часть модуля уклонения (среда, обучение, оценка и демонстрация пользуются одним кодом):
  * Perception: стереокамера -> карта глубины для сети и обстановка для предохранителя
    (только бортовые датчики: карта глубины левой камеры, ориентация и скорость по IMU);
    земля отсекается по высоте точки, при потере изображения (засветка) обстановка до HOLD_MAX
    прогнозируется по IMU из последнего облака точек, дальше модуль считается слепым;
  * наблюдение сети (26 навигационных признаков + обратная карта глубины 40x64), кодирование действий;
  * Fallback: безопасное действие вне сети (набор высоты, облет, возврат);
  * Executor: уставки -> предохранитель -> ограничение крена -> sim.step, сторожевой таймер реакции;
  * Baseline: базовый навигатор (учитель для BC и точка сравнения; видит лучи симулятора);
  * route_scene: сцены методики итоговых испытаний ТЗ (5 типов препятствий на маршруте);
  * геометрия препятствий для награды и метрик, загрузка сети, обертка AIPilot.

Модуль не импортирует mujoco и torch на верхнем уровне: его можно подключать до настройки MUJOCO_GL.
"""

import math
import os
import time
from collections import deque

import numpy as np

D = math.pi / 180.0
# масштаб действий сети (совпадает с весами)
GAMMA_MAX, BETA_MAX, ELEV_MAX = 45 * D, 10 * D, 15 * D
V_MIN, V_MAX = 9.5, 20.0
NAV_DIM, DEPTH_SHAPE, ACT_DIM = 26, (40, 64), 6
GOAL_RADIUS, DT = 30.0, 0.1

# ТЗ, таблица п. 5: фактический крен не более 30 град; уставка 27 град оставляет запас на заброс контура крена
BANK_LIMIT = 20 * D
BANK_RATE = 30 * D * DT          # изменение уставки крена за шаг 0,1 с (плавность)
BANK_RATE_EM = 60 * D * DT       # то же при экстренном уходе от столкновения
BANK_EM = 20 * D                 # крен экстренного ухода: быстрое нарастание дает больший заброс

OVERFLY_MARGIN = 5.0             # запас к горизонтальному контуру препятствия, м
OVERFLY_BAND = 15.0              # препятствие мешает эшелону цели, если его вершина выше (высота цели - 15 м)
_SPHERE, _ELLIPSOID, _CYLINDER = 2, 4, 5   # mjtGeom
AZ = np.linspace(-45, 45, 15)    # азимуты лучей sim.rays(), град; az < 0 слева


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------- восприятие ----------------
class Perception:
    """
    Карта глубины левой камеры (z-глубина, м, inf: нет данных) -> облако точек в мире по ориентации IMU.
    Точки препятствий помнятся MEM_FRAMES кадров (5 с): препятствие, ушедшее из поля зрения вбок, остается
    в обстановке, и разворот в его сторону виден как угроза.
    Обстановка в полосе высот полета (+-DZ_BAND):
      front  дальность до препятствия в коридоре шириной 2*W_CORR вдоль прогнозируемой траектории
             (дуга разворота по текущему крену и скорости; MIN_PIX точек подтверждают препятствие:
             порог чувствительности против ложных срабатываний на одиночные шумовые точки);
      left, right  лучший просвет в боковых секторах SIDE_AZ (есть ли куда отвернуть);
      near_l, near_r  ближайшее препятствие сбоку в секторах NEAR_AZ с учетом памяти (куда отворачивать нельзя);
      top    превышение самой высокой точки препятствия в коридоре (ближе D_TOP) над планером;
             inf, если препятствие упирается в верхний край кадра и вершина не видна;
      threat препятствие в коридоре ближе D_DETECT и ближе цели (обнаружение для сигнала и времени реакции);
             снимается, когда дальность больше D_DETECT + D_HYST (шум стереопары не дает мигания);
      lost   кадр потерян, blind: потерян дольше HOLD_MAX, прогноз уже недостоверен.
    """
    TAN_H, TAN_V = 0.00384 / 2 / 0.0028, 0.0024 / 2 / 0.0028          # из sensorsize и focal камеры
    CAM_POS = np.array([0.116, 0.649109, 0.111018])                   # левая камера в связанных осях
    CAM_X, CAM_Y = np.array([0.0, -1.0, 0.0]), np.array([0.0871557, 0.0, 0.996195])   # xyaxes камеры
    D_MAX, GROUND_Z, MIN_PIX = 400.0, 3.0, 3
    D_DETECT, D_HYST, D_TOP, W_CORR, DZ_BAND = 150.0, 25.0, 120.0, 30.0, 12.0
    SIDE_AZ, NEAR_AZ, HOLD_MAX, LOST_H, MEM_FRAMES = (15.0, 34.0), (15.0, 90.0), 1.0, 150.0, 20

    def __init__(self, shape=DEPTH_SHAPE):
        h, w = shape
        X, Y = np.meshgrid(((np.arange(w) + 0.5) / w * 2 - 1) * self.TAN_H, (1 - (np.arange(h) + 0.5) / h * 2) * self.TAN_V)
        self.cam_z = np.cross(self.CAM_X, self.CAM_Y)                 # камера смотрит вдоль -z
        self.dirs = X[..., None] * self.CAM_X + Y[..., None] * self.CAM_Y - self.cam_z   # луч на 1 м z-глубины
        self.shape, self.flip, self.checked = shape, False, False
        self.side_bins = np.linspace(*self.SIDE_AZ, 9)
        self.reset()

    def reset(self):
        self.cloud, self.age, self.threat = np.zeros((0, 3)), 0.0, False
        self.mem = deque(maxlen=self.MEM_FRAMES)

    def _check_orientation(self, dm, s):
        """Земля должна быть внизу кадра: на малой высоте в горизонтальном полете нижние строки заполнены."""
        if self.checked or s["H"] > self.LOST_H or abs(s["gamma"]) > 5 * D or abs(s["theta"]) > 5 * D:
            return
        bottom, top = np.isfinite(dm[-3:]).mean(), np.isfinite(dm[:3]).mean()
        if bottom > 0.5 or top > 0.5:
            self.flip, self.checked = top > bottom, True
            if self.flip:
                print("[восприятие] карта глубины перевернута по вертикали, включен разворот строк")

    def _project(self, pos, R):
        """Прогноз карты глубины по IMU: облако последнего кадра переносится в текущее положение камеры."""
        h, w = self.shape
        b = (self.cloud - pos) @ R - self.CAM_POS
        z = -(b @ self.cam_z)
        u, v = b @ self.CAM_X / np.maximum(z, 1e-3) / self.TAN_H, b @ self.CAM_Y / np.maximum(z, 1e-3) / self.TAN_V
        c, r = np.floor((u + 1) / 2 * w).astype(int), np.floor((1 - v) / 2 * h).astype(int)
        ok = (z > 1.0) & (c >= 0) & (c < w) & (r >= 0) & (r < h)
        dm = np.full(self.shape, np.inf)
        np.minimum.at(dm, (r[ok], c[ok]), z[ok])
        p = np.pad(dm, 1, constant_values=np.inf)                    # закрыть дыры от приближения
        return np.minimum(dm, np.min([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0))

    def update(self, sim, s, blackout=False):
        import motoglider as mg
        dm = None if blackout else sim.depth()
        t0 = time.perf_counter()
        pos, R = s["pos"], mg.matrix(s["gamma"], s["theta"], s["heading"])
        if dm is not None:
            self._check_orientation(dm, s)
            dm = dm[::-1] if self.flip else dm
        lost = dm is None or (not np.isfinite(dm).any() and s["H"] < self.LOST_H)
        top = np.zeros(0, bool)
        if lost:
            self.age += DT
            dm = self._project(pos, R) if len(self.cloud) else np.full(self.shape, np.inf)
            self.mem.append(np.zeros((0, 3)))
        else:
            ok = np.isfinite(dm)
            self.cloud, self.age = pos + (self.CAM_POS + dm[..., None] * self.dirs)[ok] @ R.T, 0.0
            high = self.cloud[:, 2] > self.GROUND_Z
            mem_pts = self.cloud[high]
            top_pts = (np.nonzero(ok)[0] == 0)[high]
            if len(mem_pts) > 500:
                self.mem.append(mem_pts[::2])
                top = top_pts[::2]
            else:
                self.mem.append(mem_pts)
                top = top_pts
        P = np.concatenate([self.mem[-1], *list(self.mem)[:-1]])
        pad = max(0, len(P) - len(top))
        out = self._situation(s, P, np.r_[top[:len(P)], np.zeros(pad, bool)])
        inv = np.where(np.isfinite(dm), np.clip(20.0 / np.maximum(dm, 1.0), 0.0, 1.0), 0.0).astype(np.float32)
        out.update(inv=inv, lost=lost, blind=lost and (not len(self.cloud) or self.age > self.HOLD_MAX),
                   t_proc=time.perf_counter() - t0)
        return out

    def _situation(self, s, P, top_row):
        """P: точки препятствий в мире (текущий кадр и память), top_row: точка из верхней строки кадра."""
        pos, hd = s["pos"], s["heading"]
        rel = P - pos
        f = rel[:, :2] @ np.array([math.sin(hd), math.cos(hd)])
        l = rel[:, :2] @ np.array([math.cos(hd), -math.sin(hd)])
        obst = f > 0
        band = obst & (np.abs(rel[:, 2]) <= self.DZ_BAND)
        kap = 9.81 * math.tan(s["gamma"]) / max(s["V"], 5.0) ** 2           # кривизна разворота, 1/м
        dev = np.hypot(f, l - 1.0 / kap) - 1.0 / abs(kap) if abs(kap) > 1e-4 else l
        corr = band & (np.abs(dev) <= self.W_CORR)
        k = self.MIN_PIX - 1
        rng = np.hypot(f, l)
        front = float(np.partition(rng[corr], k)[k]) if corr.sum() > k else self.D_MAX
        az = np.degrees(np.arctan2(l, f))
        side, near = [], []
        for sg in (-1.0, 1.0):
            m = band & (sg * az >= self.SIDE_AZ[0]) & (sg * az < self.SIDE_AZ[1])
            free = np.full(len(self.side_bins) - 1, self.D_MAX)
            np.minimum.at(free, np.clip(np.digitize(sg * az[m], self.side_bins) - 1, 0, len(free) - 1), rng[m])
            side.append(float(free.max()))
            n = band & (sg * az >= self.NEAR_AZ[0]) & (sg * az < self.NEAR_AZ[1])
            # поперечное расстояние: боковое препятствие видно на любой дальности впереди
            near.append(float(np.abs(l[n]).min()) if n.any() else self.D_MAX)
        up = obst & (np.abs(dev) <= self.W_CORR) & (rng < self.D_TOP)
        top = -np.inf if not up.any() else (np.inf if (up & top_row).any() else float(rel[up, 2].max()))
        goal_h = float(np.linalg.norm(s["goal"][:2] - pos[:2]))
        self.threat = front < min(self.D_DETECT + self.D_HYST * self.threat, goal_h)
        return dict(front=front, left=side[0], right=side[1], near_l=near[0], near_r=near[1], top=top, threat=self.threat)


# ---------------- наблюдение и действия ----------------
def observe(s, inv=None):
    pos, hdg, goal, sp, om = s["pos"], s["heading"], s["goal"], s["setpoints"], s["omega"]
    dv = goal[:2] - pos[:2]
    dist = float(np.linalg.norm(dv))
    berr = wrap(math.atan2(dv[0], dv[1]) - hdg)
    nav = np.array([
        s["V"] / 20.0, s["alpha"] / (15 * D), s["beta"] / (15 * D), s["gamma"] / GAMMA_MAX, s["theta"] / (30 * D),
        math.sin(hdg), math.cos(hdg), om[0] / 3.0, om[1] / 3.0, om[2] / 3.0, s["ny"] / 4.0,
        s["headwind"] / 10.0, s["crosswind"] / 10.0, dist / 2000.0, math.sin(berr), math.cos(berr),
        (goal[2] - pos[2]) / 100.0, s["H"] / 300.0,
        s["vel"][2] / 10.0, float(np.linalg.norm(s["vel"][:2])) / 20.0, s["battery"], float(s["motor_on"]),
        sp["gamma"] / GAMMA_MAX, (sp["V"] - V_MIN) / (V_MAX - V_MIN), sp["beta"] / BETA_MAX, sp["throttle"],
    ], dtype=np.float32)
    return nav if inv is None else np.concatenate([nav, inv.ravel()])


def tensors(obs, device):
    """Пакет наблюдений (B, obs_dim) -> навигационный тензор (B, 26) и карта глубины (B, 1, 40, 64)."""
    import torch
    obs = np.atleast_2d(obs)
    return (torch.as_tensor(obs[:, :NAV_DIM], device=device),
            torch.as_tensor(obs[:, NAV_DIM:].reshape(-1, 1, *DEPTH_SHAPE), device=device))


def decode(a):
    """Действие сети [-1, 1]^6 -> (крен, скорость, скольжение, газ, руль высоты или None)."""
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    return (float(a[0]) * GAMMA_MAX, V_MIN + (V_MAX - V_MIN) * 0.5 * (float(a[1]) + 1.0), float(a[2]) * BETA_MAX,
            0.5 * (float(a[3]) + 1.0), float(a[4]) * ELEV_MAX if a[5] > 0 else None)


def encode(gamma, V, beta, thr, elevator=None):
    """Уставки -> действие сети; шестой элемент: 1 экстренный режим, -1 обычный."""
    u = [gamma / GAMMA_MAX, 2.0 * (V - V_MIN) / (V_MAX - V_MIN) - 1.0, beta / BETA_MAX, 2.0 * thr - 1.0,
         (elevator or 0.0) / ELEV_MAX, 1.0 if elevator is not None else -1.0]
    return np.clip(np.array(u, dtype=np.float32), -1.0, 1.0)


# ---------------- геометрия препятствий (награда и метрики, не для бортового модуля) ----------------
def _ellipsoid_dist(y, e):
    """Точное расстояние от точки y (в первом октанте) до эллипсоида с полуосями e (корень по бисекции)."""
    if np.sum((y / e) ** 2) <= 1.0:
        return 0.0
    lo, hi = 0.0, float(e.max() * np.linalg.norm(y))
    for _ in range(60):
        t = 0.5 * (lo + hi)
        lo, hi = (t, hi) if np.sum((e * y / (t + e * e)) ** 2) > 1.0 else (lo, t)
    return float(np.linalg.norm(y - e * e * y / (lo + e * e)))


def obstacle_info(sim, pos, level=None):
    """
    Расстояние от центра планера до поверхности ближайшего препятствия и признак пролета над препятствием. Пролетом сверху (запрещен сети) считается
    нахождение над контуром препятствия, вершина которого поднимается в эшелон цели level;
    низкая застройка ниже эшелона не считается. level=None: любое препятствие под планером.
    """
    d_min, over = 1e3, False
    for o in sim.obstacles():
        rel = np.abs(np.asarray(pos) - o["pos"])
        sz, t = o["size"], o["type"]
        if t in (_SPHERE, _CYLINDER):
            hz = sz[0] if t == _SPHERE else sz[1]
            horiz = max(0.0, math.hypot(rel[0], rel[1]) - sz[0])
            d = float(np.linalg.norm(rel)) - sz[0] if t == _SPHERE else math.hypot(horiz, max(0.0, rel[2] - hz))
            inside = horiz <= OVERFLY_MARGIN
        else:
            hz = sz[2]
            ex = np.maximum(rel - sz[:3], 0.0)
            d = float(np.linalg.norm(ex))
            if t == _ELLIPSOID:
                d = _ellipsoid_dist(rel, sz[:3])
            inside = ex[0] <= OVERFLY_MARGIN and ex[1] <= OVERFLY_MARGIN
        top = o["pos"][2] + hz
        over |= bool(inside and pos[2] >= top and (level is None or top > level - OVERFLY_BAND))
        d_min = min(d_min, d)
    return d_min, over


# ---------------- базовый навигатор ----------------
class Baseline:
    """
    Навигатор без сети: курс на цель, высота цели, обход по лучам симулятора (-5 и +5 град).
    Если в поле лучей (+-45 град) препятствие ближе REACH, выбирается ближайшее к цели направление, где
    коридор шириной 2*HALF_W свободен, и крен ставится пропорционально его азимуту (плавный облет
    кромки, без рысканья); сторона облета держится, пока на ней есть просвет.
    avoid=False: без обхода (ручной режим демонстрации).
    """
    REACH, HALF_W, GAIN = 170.0, 15.0, 1.2

    def __init__(self, avoid=True):
        self.use_avoid = avoid
        self.reset()

    def reset(self):
        self.motor, self.avoid = False, 0.0

    def __call__(self, sim, s, per=None):
        goal, pos = s["goal"], s["pos"]
        dv = goal[:2] - pos[:2]
        dist = float(np.linalg.norm(dv))
        err = wrap(math.atan2(dv[0], dv[1]) - s["heading"])
        gamma = float(np.clip(err, -BANK_LIMIT, BANK_LIMIT))
        col = sim.rays().reshape(5, 15)[2:4].min(axis=0) * 300.0 if self.use_avoid else None
        reach = min(self.REACH, dist)
        if col is not None and dist > 60 and (col < reach).any():
            near = col < reach                                              # луч k закрывает направление j,
            half = np.degrees(np.arctan2(self.HALF_W, np.maximum(col, 1.0)))  # если ближе своей полуширины
            ok = ~(near[None, :] & (np.abs(AZ[:, None] - AZ[None, :]) <= half[None, :])).any(axis=1)
            side = ok & (np.sign(AZ) != -self.avoid)
            cand = AZ[side if side.any() else ok]
            if len(cand):
                az = float(cand[np.argmin(np.abs(cand - math.degrees(err)))])
                self.avoid = self.avoid or float(np.sign(az))
                gamma = float(np.clip(self.GAIN * az * D, -BANK_LIMIT, BANK_LIMIT))
            else:
                self.avoid = self.avoid or (-1.0 if col[AZ < 0].mean() > col[AZ > 0].mean() else 1.0)
                gamma = self.avoid * BANK_LIMIT
        else:
            self.avoid = 0.0
        dz = pos[2] - goal[2]                           # держим высоту цели, мотор с гистерезисом
        if dz < -6.0:
            self.motor = True
        elif dz > 3.0:
            self.motor = False
        V = 13.0 if self.motor else 12.0 + (2.0 if dz > 10.0 else 0.0) + 0.5 * max(s["headwind"], 0.0)
        return gamma, V, 0.0, 1.0 if self.motor else 0.0


# ---------------- предохранитель ----------------
class Fallback:
    """
    Безопасное действие вне сети (ТЗ: облет невозможен -> набор высоты или возврат; модель не уверена ->
    набор высоты). Режимы и условия (front: дальность по прогнозируемой траектории):
      набор     высота меньше H_MIN или модуль слеп (изображение потеряно дольше Perception.HOLD_MAX);
      уход      до столкновения меньше T_CRIT секунд (или ближе D_CRIT): крен BANK_EM в свободную
                сторону (сторона без близких препятствий сбоку), полный газ, а если вершина препятствия не выше PULL_TOP, импульс руля высоты
                на кабрирование (экстренный режим);
      облет     сеть не начала маневр за отведенное время (late) и сбоку есть просвет: крен в его сторону;
      набор или возврат  препятствие ближе D_TRIG и просвета нет ни слева, ни справа (в боковом секторе
                свободно меньше min(D_SIDE, front + GAP_GAIN), поворот ничего не дает, как у стены), или сеть
                не уверена; возврат разворотом, если вершину не набрать до препятствия (CLIMB_RATE
                с запасом CLIMB_MARGIN) или она не видна.
    Выключается, когда впереди чисто HOLD секунд. Пролет над препятствием допускается только здесь.
    """
    D_TRIG, D_SIDE, GAP_GAIN, D_CRIT, T_CRIT, H_MIN, CLEAR, HOLD = 150.0, 180.0, 40.0, 40.0, 5.0, 15.0, 180.0, 3.0
    PULL_STEPS, PULL_ELEV, PULL_TOP, V_CLIMB, CLIMB_RATE, CLIMB_MARGIN = 5, -8 * D, 8.0, 12.5, 2.0, 10.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode, self.timer, self.pull, self.side = None, 0.0, 0, 0.0

    def _choose(self, s, p, unsure, late, gap):
        front = p["front"]
        if s["H"] < self.H_MIN or p["blind"]:
            return "набор"
        if front < max(self.D_CRIT, self.T_CRIT * max(s["V"], 5.0)):
            return "уход"
        if late and gap:
            return "облет"
        if front < self.D_TRIG and (unsure or not gap or late):
            return "возврат" if p["top"] + self.CLIMB_MARGIN > self.CLIMB_RATE * front / max(s["V"], 5.0) else "набор"
        return None

    def __call__(self, s, p, sp, unsure=False, late=False):
        gap = max(p["left"], p["right"]) >= min(self.D_SIDE, p["front"] + self.GAP_GAIN)
        mode = self._choose(s, p, unsure, late, gap)
        if mode:
            if self.mode is None or (mode == "уход" and self.mode != "уход"):
                self.side = -1.0 if (p["near_l"], p["left"]) > (p["near_r"], p["right"]) else 1.0
                self.pull = self.PULL_STEPS if mode == "набор" or (mode == "уход" and p["top"] <= self.PULL_TOP) else 0
                self.mode = mode
            self.timer = self.HOLD
        elif self.mode and p["front"] > self.CLEAR:
            self.timer -= DT
            self.mode = self.mode if self.timer > 0.0 else None
        if self.mode is None:
            return None
        gamma, V, _, thr = sp[:4]
        elev = self.PULL_ELEV if self.pull else None
        self.pull = max(0, self.pull - 1)
        if self.mode == "облет":
            return self.side * BANK_LIMIT, V, 0.0, thr, None
        if self.mode in ("уход", "возврат"):
            return self.side * (BANK_EM if self.mode == "уход" else BANK_LIMIT), self.V_CLIMB, 0.0, 1.0, elev
        return float(np.clip(gamma, -20 * D, 20 * D)), self.V_CLIMB, 0.0, 1.0, elev


# ---------------- исполнитель ----------------
class Executor:
    """
    Уставки политики -> предохранитель -> ограничение и сглаживание крена -> sim.step.
    Сторожевой таймер реакции (ТЗ: не более 0,3 с от обнаружения до маневра): если после обнаружения
    препятствия политика за REACT_STEPS шагов не начала маневр (крен изменился на MAN_BANK или
    больше, экстренный режим, прибавка газа), маневр начинает предохранитель.
    reactions: времена реакции, мс (шаги с момента обнаружения плюс время обработки кадра).
    """
    REACT_STEPS, MAN_BANK, MAN_THR = 2, 10 * D, 0.5

    def __init__(self, fallback=True):
        self.fb = Fallback() if fallback else None
        self.reset()

    def reset(self):
        if self.fb:
            self.fb.reset()
        self.k, self.onset, self.reactions = 0, None, []

    def __call__(self, sim, s, sp, p, unsure=False):
        sp = (tuple(sp) + (None,))[:5]
        # замедление при близком препятствии — критично для захвата цели BC
        if p["front"] < 150.0 or min(p["near_l"], p["near_r"]) < 60.0:
            sp = (sp[0], min(sp[1], 11.0), sp[2], sp[3], sp[4])
        prev, late = s["setpoints"], False
        if p["threat"]:
            if self.onset is None:
                self.onset = [self.k, prev["gamma"], prev["throttle"], False]
            k0, g0, t0, done = self.onset
            moved = (abs(sp[0] - g0) >= self.MAN_BANK or abs(sp[0]) >= 10 * D or sp[4] is not None
                     or sp[3] - t0 >= self.MAN_THR)
            late = not done and not moved and self.k - k0 >= self.REACT_STEPS and self.fb is not None
        else:
            self.onset = None
        fb = self.fb(s, p, sp, unsure, late) if self.fb else None
        if self.onset and not self.onset[3] and (moved or fb):
            self.onset[3] = True
            self.reactions.append((self.k - self.onset[0]) * DT * 1000 + p["t_proc"] * 1000)
        gamma, V, beta, thr, elev = fb or sp
        g, rate = prev["gamma"], BANK_RATE_EM if fb and self.fb.mode == "уход" else BANK_RATE
        gamma = float(np.clip(np.clip(gamma, g - rate, g + rate), -BANK_LIMIT, BANK_LIMIT))
        self.k += 1
        return sim.step(gamma, V, beta, thr, elevator=elev), fb is not None


# ---------------- сцены методики итоговых испытаний ----------------
ROUTE_KINDS = ("дерево", "группа деревьев", "здание", "ЛЭП", "скала")

def route_layout(rng, kind, L):
    """Препятствия на маршруте: (геометрия, вдоль маршрута, поперек вправо, масштаб, высота центра или None)."""
    x, y, u = rng.uniform(0.4, 0.6) * L, rng.uniform(-8.0, 8.0), rng.uniform
    if kind == "дерево":
        return [("obs_tree", x, y, u(2.4, 2.8), None)]
    if kind == "группа деревьев":
        return [("obs_tree", x, y, u(2.2, 2.5), None), ("obs_tower_1", x + u(20, 35), y - u(12, 25), u(0.9, 1.1), None),
                ("obs_tower_3", x - u(15, 30), y + u(12, 25), u(0.9, 1.1), None)]
    if kind == "здание":
        return [("obs_building", x, y, u(1.5, 1.8), None)]
    if kind == "ЛЭП":
        return [("obs_tower_1", x, y, u(1.0, 1.2), None), ("obs_tower_2", x, y - u(45, 60), 1.0, None),
                ("obs_tower_3", x, y + u(45, 60), 1.0, None)]
    s = u(2.5, 3.0)                                     # скала: шар, наполовину в земле
    return [("obs_aerostat_static", x, y, s, 6.0 * s)]


def route_scene(sim, rng, kind=None, H=50.0, L=600.0, V=15.0, wind=(0.0, 0.0), bearing=90.0):
    """
    Сцена методики ТЗ: старт на высоте H со скоростью V, цель на той же высоте в L м по курсу bearing,
    одно препятствие типа kind на маршруте, остальные препятствия убраны. Возвращает (состояние, тип).
    """
    import mujoco
    m, base = sim.model, sim._base
    kind = kind or ROUTE_KINDS[int(rng.integers(len(ROUTE_KINDS)))]
    sim.default_scene()
    sim.hide_obstacle()
    idx = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): k for k, g in enumerate(sim.static)}
    b = bearing * D
    fwd, right = np.array([math.sin(b), math.cos(b)]), np.array([math.cos(b), -math.sin(b)])
    sim.scene_pos[:] = [[5000.0 + 100.0 * k, 5000.0, -500.0] for k in range(len(sim.static))]
    for name, a, c, sc, z in route_layout(rng, kind, L):
        k = idx[name]
        g = sim.static[k]
        m.geom_size[g], m.geom_rbound[g], m.geom_aabb[g] = base["size"][k] * sc, base["rbound"][k] * sc, base["aabb"][k] * sc
        m.bvh_aabb[m.body_bvhadr[sim.static_body[k]]] = base["bvh"][k] * sc
        sz, t = m.geom_size[g], m.geom_type[g]
        z = z if z is not None else (sz[1] if t == _CYLINDER else sz[0] if t == _SPHERE else sz[2])
        sim.scene_pos[k] = [*(fwd * a + right * c), z]
    for i, mv in enumerate(sim.movers):
        mv["p0"] = np.array([-5000.0, -5000.0 - 200.0 * i, -500.0])
    goal = (fwd[0] * L, fwd[1] * L, H)
    return sim.reset(start=(0.0, 0.0, H), heading_deg=bearing, airspeed=V, wind=wind, goal=goal), kind

# ---------------- сеть ----------------
def device_of(cpu=False):
    import torch
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")

def load_net(path, device):
    """Создает ActorCritic; если файл есть, загружает веса. Возвращает (сеть, загружено ли)."""
    import torch
    from ai_pilot import ActorCritic
    net = ActorCritic(nav_dim=NAV_DIM, depth_shape=DEPTH_SHAPE, use_depth=True).to(device)
    if not path or not os.path.isfile(path):
        return net, False
    ck = torch.load(path, map_location=device)
    net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    return net, True


class AIPilot:
    """
    Обученная сеть как политика (детерминированно), держит скрытое состояние GRU.
    unsure: вероятность экстренного режима в зоне UNSURE, сеть не решила, как уходить от препятствия.
    """
    UNSURE = (0.3, 0.7)

    def __init__(self, path, cpu=False):
        import torch
        self.torch, self.device = torch, device_of(cpu)
        self.net, ok = load_net(path, self.device)
        if not ok:
            raise FileNotFoundError(path)
        self.net.eval()
        self.reset()

    def reset(self):
        self.h, self.unsure = self.net.initial_hidden(1, self.device), False

    def __call__(self, sim, s, per):
        from ai_pilot import RAW_MAX
        nav, dep = tensors(observe(s, per["inv"]), self.device)
        with self.torch.no_grad():
            mean, logit, _, _, self.h = self.net(nav, dep, self.h)
        p = float(self.torch.sigmoid(logit[0]))
        self.unsure = self.UNSURE[0] < p < self.UNSURE[1]
        u = self.torch.tanh(mean[0].clamp(-RAW_MAX, RAW_MAX)).cpu().numpy()
        return decode(np.append(u, 1.0 if p > 0.5 else -1.0))
