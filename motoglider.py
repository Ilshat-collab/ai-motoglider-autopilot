"""
Физика мотопланера для MuJoCo. Все характеристики аппарата берутся из
motoglider.xml (раздел custom), поэтому модель и код всегда согласованы.

Что делает модуль:
  * аэродинамические силы и моменты по коэффициентам (обозначения ГОСТ 20058-80,
    структура модели по R. W. Beard, T. W. McLain, Small Unmanned Aircraft, 2012);
  * тяга складного винта и потребляемая мощность, расход батареи;
  * четыре канала внутреннего управления (ПИД): крен -> элероны, воздушная скорость ->
    тангаж -> руль высоты, скольжение -> руль направления, газ -> мотор;
  * экстренный режим: прямое управление рулем высоты при внезапном препятствии;
  * постоянный горизонтальный ветер;
  * статичные и движущиеся препятствия (аэростаты, дирижабль), внезапное препятствие,
    случайные сцены для обучения на незнакомой обстановке;
  * карта глубины левой камеры с ошибками стереопары;
  * обнаружение столкновений и разрушения.

Системы координат:
  мир MuJoCo:            x на восток, y на север, z вверх;
  связанная MuJoCo:      x вперед, y влево, z вверх;
  связанная по ГОСТ:     X вперед, Y вверх, Z вправо (X = x, Y = z, Z = -y).
Знаки рулей по ГОСТ 20058-80: положительное отклонение дает отрицательный момент.
  Элероны: правый вниз, левый вверх (крен влево). Руль высоты: задняя кромка вниз
  (пикирование). Руль направления: задняя кромка вправо (нос вправо).
"""

import math
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DEG = math.pi / 180.0
G0 = 9.81
GOST = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # связанные оси MuJoCo -> ГОСТ
SURF = ("ail_l", "ail_r", "elevator", "rudder")
MOVE_KIND = {1: "привязной аэростат", 2: "свободный аэростат", 3: "дирижабль"}


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


_T = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_F = np.diag([1.0, -1.0, -1.0])


def attitude(R):
    """Крен, тангаж (рад) и курс (рад, от севера по часовой) по матрице поворота корпуса."""
    M = _T @ R @ _F
    return (math.atan2(M[2, 1], M[2, 2]), -math.asin(max(-1.0, min(1.0, M[2, 0]))), math.atan2(M[1, 0], M[0, 0]))


def matrix(roll, pitch, heading):
    return _T @ (_rot("z", heading) @ _rot("y", pitch) @ _rot("x", roll)) @ _F


class GliderSim:
    """
    Модель мотопланера.

    Основные методы:
      reset(...)                         начало эпизода, возвращает state();
      step(gamma, V, beta, throttle)     шаг 0,1 с с уставками внутренним контурам:
                                         крен (рад), скорость (м/с), скольжение (рад), газ 0...1;
      state(), wind(), obstacles(), rays(), depth(), set_goal(xyz).
    """

    def __init__(self, xml_path=None, seed=0, depth=False, depth_size=(64, 40)):
        self.model = mujoco.MjModel.from_xml_path(xml_path or os.path.join(HERE, "motoglider.xml"))
        self.data = mujoco.MjData(self.model)
        m = self.model
        n = lambda name: np.array(m.numeric(name).data, dtype=float)  # noqa: E731
        self.S, self.l, self.bA, self.lam = n("glider_geom")
        (self.cya0, self.cya_a, self.M, self.a0, self.cya_wz, self.cya_dv) = n("aero_lift")
        self.cx0, self.e = n("aero_drag")
        self.cz_b, self.cz_dn = n("aero_side")
        self.mx = n("aero_roll")
        self.my = n("aero_yaw")
        self.mz0, self.mz_a, self.mz_wz, self.mz_dv = n("aero_pitch")
        self.Jy = float(m.body_inertia[m.body("glider").id][2])   # момент инерции рыскания (ось Y по ГОСТ)
        self.rho0 = n("atmos")[0]
        (self.Sv, self.Cv, self.km, self.eta, self.dead, self.tau, self.unfold, _) = n("prop")
        self.E_total, self.P_av = n("battery")
        self.V_NE, self.n_max, self.kV, self.kn = n("limits")
        self.k_roll, self.k_pitch, self.k_yaw = n("pid_roll"), n("pid_pitch"), n("pid_yaw")
        self.V0, self.kmin, self.kmax, rate, f_ctrl = n("pid_sched")
        self.rate = rate * DEG
        self.lim = n("surf_limits") * DEG
        sl = n("setpoint_limits")
        self.sp_lim = (sl[0] * DEG, sl[1] * DEG, sl[2], sl[3])   # крен, скольжение (рад); скорость мин, макс (м/с)
        st = n("stereo")
        self.f_px, self.B, self.sig_d, self.d_min, self.D_max, self.flex, self.mis, self.resid = st
        self.r_s = n("wind_sensor")
        self.rng = np.random.default_rng(seed)
        self.dt = m.opt.timestep
        self.n_sub = int(round(0.1 / self.dt))
        self.n_ctrl = max(1, int(round(1.0 / (f_ctrl * self.dt))))
        self.bid = m.body("glider").id
        self.act = np.array([m.actuator(s).id for s in SURF])
        self.qadr = np.array([m.jnt_qposadr[m.joint(s).id] for s in SURF])
        self.goal_id = m.body_mocapid[m.body("goal").id]
        self.ground = m.geom("ground").id
        self.prop_geom = m.geom("prop_disk").id
        self.sudden_id = m.body_mocapid[m.body("sudden_obstacle").id]
        self.sudden_geom = m.geom("sudden_obstacle_g").id
        self.sudden_active = False
        self.emergency = False
        self.movers = []
        for i in range(m.nnumeric):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, i)
            if name.startswith("move_"):
                body = m.body(name[5:])
                d = n(name)
                g = m.geom(name[5:] + "_g")
                self.movers.append(dict(name=name[5:], mocap=m.body_mocapid[body.id], geom=g.id, kind=int(d[0]),
                                        A=d[1:4].copy(), T=float(d[4]), phase0=float(d[5]), v0=d[6:9].copy(),
                                        p0=m.body_pos[body.id].copy(), phase=float(d[5]), v=d[6:9].copy()))
        self.mover_geoms = {mv["geom"]: mv["name"] for mv in self.movers}
        self.static = [i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obs_")]
        # каждое препятствие отдельное тело mocap: его можно переносить и менять размер без потери столкновений
        self.static_body = [int(m.geom_bodyid[g]) for g in self.static]
        self.static_mocap = [int(m.body_mocapid[b]) for b in self.static_body]
        self._base = dict(pos=m.body_pos[self.static_body].copy(), size=m.geom_size[self.static].copy(),
                          rbound=m.geom_rbound[self.static].copy(), aabb=m.geom_aabb[self.static].copy(),
                          bvh=m.bvh_aabb[m.body_bvhadr[self.static_body]].copy(),
                          movers=[(mv["p0"].copy(), mv["phase0"], mv["v0"].copy()) for mv in self.movers])
        self.scene_pos = self._base["pos"].copy()
        self.ray_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        az = np.linspace(-45, 45, 15) * DEG
        el = np.linspace(-20, 20, 5) * DEG - 5 * DEG
        A, E = np.meshgrid(az, el)
        self.ray_dirs = np.stack([np.cos(E) * np.cos(A), -np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.renderer = None
        self.depth_w, self.depth_h = depth_size
        if depth:
            self.renderer = mujoco.Renderer(m, self.depth_h, self.depth_w)
        self.scene_opt = mujoco.MjvOption()
        self.scene_opt.geomgroup[:] = 0
        self.scene_opt.geomgroup[0] = 1
        self.zfar = m.vis.map.zfar * m.stat.extent
        self.bad = int(mujoco.mjtWarning.mjWARN_BADQACC)
        self.reset()

    # ---------------- сброс ----------------
    def reset(self, start=(0.0, 0.0, 130.0), heading_deg=90.0, airspeed=12.0, wind=(0.0, 0.0),
              goal=(1600.0, 0.0, 110.0), randomize_obstacles=False, battery=1.0):
        """wind = (скорость, м/с; откуда дует, град по компасу); постоянный горизонтальный ветер."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.mocap_pos[self.static_mocap] = self.scene_pos
        spd, frm = wind
        self.W = -float(spd) * np.array([math.sin(frm * DEG), math.cos(frm * DEG), 0.0])
        for mv in self.movers:
            mv["phase"] = self.rng.uniform(0, 2 * math.pi) if randomize_obstacles else mv["phase0"]
            mv["v"] = mv["v0"] * (self.rng.uniform(0.5, 1.5) if randomize_obstacles else 1.0) + (self.W if mv["kind"] == 2 else 0.0)
        R0 = matrix(0.0, 0.0, heading_deg * DEG)
        d.qpos[0:3] = start
        mujoco.mju_mat2Quat(d.qpos[3:7], R0.reshape(9))
        V0 = float(airspeed)
        cya = 2 * self._mass() * G0 / (self._rho(start[2]) * self.S * V0 * V0)
        a = (cya - self.cya0) / self.cya_a
        d.qvel[0:3] = self.W + V0 * (R0[:, 0] * math.cos(a) - R0[:, 2] * math.sin(a))
        d.mocap_pos[self.goal_id] = goal
        d.mocap_pos[self.sudden_id] = [0.0, 0.0, -2000.0]
        self.sudden_active = False
        self.emergency = False
        self.blend, self.dv_em = 0.0, 0.0
        self.ramp, self.th_switch = 0.0, 0.0
        self.t = 0.0
        self.sp = dict(gamma=0.0, V=V0, beta=0.0, throttle=0.0, elevator=None)
        self.i_g = self.i_v = self.i_b = 0.0
        self.th_cmd = 0.0
        self.cmd = np.zeros(3)
        self.level, self.motor_on, self.on_time = 0.0, False, 0.0
        self.E = self.E_total * battery
        self.E_motor = 0.0
        self.P = self.N = 0.0
        self.crashed, self.reason = False, ""
        self.aero = dict(V=V0, alpha=a, beta=0.0, cya=cya, cxa=0.0, ny=1.0)
        self._move_obstacles(0.0)
        mujoco.mj_forward(m, d)
        R = self._kin()[1]
        self.lever = R.T @ (d.subtree_com[self.bid] - d.xipos[self.bid])
        self.warn0 = d.warning[self.bad].number
        return self.state()

    def set_goal(self, xyz):
        self.data.mocap_pos[self.goal_id] = np.asarray(xyz, dtype=float)

    # ---------------- служебное ----------------
    def _mass(self):
        return float(self.model.body_subtreemass[self.bid])

    def _rho(self, H):
        return self.rho0 * (1.0 - 2.2558e-5 * max(H, 0.0)) ** 4.2559          # ГОСТ 4401-81, тропосфера

    def _kin(self):
        d = self.data
        R9 = np.empty(9)
        mujoco.mju_quat2Mat(R9, d.qpos[3:7])
        return d.qpos[0:3].copy(), R9.reshape(3, 3), d.qvel[0:3].copy(), d.qvel[3:6].copy()

    def _deflections(self):
        q = self.data.qpos[self.qadr]
        return 0.5 * (q[0] - q[1]), -q[2], q[3]          # delta_e (элероны), delta_v (РВ), delta_n (РН)

    def mover_state(self, mv, t):
        """Положение и скорость движущегося препятствия в момент t (формулы документа)."""
        A, T, ph = mv["A"], mv["T"], mv["phase"]
        W = 2 * math.pi / T
        x = W * t + ph
        if mv["kind"] == 1:
            off = np.array([A[0] * math.sin(x), A[1] * math.cos(x), A[2] * math.sin(2 * x)])
            vel = np.array([A[0] * W * math.cos(x), -A[1] * W * math.sin(x), 2 * A[2] * W * math.cos(2 * x)])
        elif mv["kind"] == 2:
            off = mv["v"] * t + np.array([0.0, 0.0, A[2] * math.sin(x)])
            vel = mv["v"] + np.array([0.0, 0.0, A[2] * W * math.cos(x)])
        else:
            off = A * (2 / math.pi) * math.asin(math.sin(x))
            vel = A * (2 / math.pi) * W * math.copysign(1.0, math.cos(x))
        return mv["p0"] + off, vel

    def _move_obstacles(self, t):
        for mv in self.movers:
            self.data.mocap_pos[mv["mocap"]] = self.mover_state(mv, t)[0]

    # ---------------- аэродинамика (ГОСТ 20058-80) ----------------
    def forces(self, Vg, wg, defl, rho):
        """Vg, wg: скорость относительно воздуха и угловая скорость в осях ГОСТ. Возвращает силу, момент (оси ГОСТ)."""
        Vx, Vy, Vz = Vg
        V = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
        if V < 0.5:
            return np.zeros(3), np.zeros(3), dict(V=V, alpha=0.0, beta=0.0, cya=0.0, cxa=0.0)
        alpha = math.atan2(-Vy, Vx)
        beta = math.asin(max(-1.0, min(1.0, Vz / V)))
        q = 0.5 * rho * V * V
        wx = wg[0] * self.l / (2 * V)
        wy = wg[1] * self.l / (2 * V)
        wz = wg[2] * self.bA / V
        de, dv, dn = defl
        e1 = math.exp(max(-200.0, min(200.0, -self.M * (alpha - self.a0))))
        e2 = math.exp(max(-200.0, min(200.0, self.M * (alpha + self.a0))))
        sig = (1 + e1 + e2) / ((1 + e1) * (1 + e2))
        lin = self.cya0 + self.cya_a * alpha
        sa = math.sin(alpha)
        cya = (1 - sig) * lin + sig * 2 * math.copysign(1.0, alpha) * sa * sa * math.cos(alpha) + self.cya_wz * wz + self.cya_dv * dv
        cxa = self.cx0 + lin * lin / (math.pi * self.e * self.lam)
        cz = self.cz_b * beta + self.cz_dn * dn
        mx = self.mx[0] * beta + self.mx[1] * wx + self.mx[2] * wy + self.mx[3] * de + self.mx[4] * dn
        my = self.my[0] * beta + self.my[1] * wx + self.my[2] * wy + self.my[3] * de + self.my[4] * dn
        mz = self.mz0 + self.mz_a * alpha + self.mz_wz * wz + self.mz_dv * dv
        qS = q * self.S
        ca, sa_ = math.cos(alpha), math.sin(alpha)
        F = np.array([-cxa * qS * ca + cya * qS * sa_, cxa * qS * sa_ + cya * qS * ca, cz * qS])
        Mo = np.array([mx * qS * self.l, my * qS * self.l, mz * qS * self.bA])
        return F, Mo, dict(V=V, alpha=alpha, beta=beta, cya=cya, cxa=cxa)

    def thrust(self, level, V, rho):
        """Тяга по модели Beard, McLain; при выключенном моторе винт сложен и тяга равна нулю."""
        if level <= 0.0:
            return 0.0, 0.0
        P = 0.5 * rho * self.Sv * self.Cv * ((self.km * level) ** 2 - V * V)
        if P <= 0.0:
            return 0.0, 0.0
        vi = -V / 2 + math.sqrt(V * V / 4 + P / (2 * rho * self.Sv))      # импульсная теория винта
        return P, P * (V + vi) / self.eta

    # ---------------- внутренние контуры ----------------
    def _pid(self, dtc):
        st = self._attitude_now()
        g, th, b, wx, wy, wz, V = st
        ks = min(self.kmax, max(self.kmin, (self.V0 / max(V, 5.0)) ** 2))
        kg, kig, kwx = self.k_roll
        kt, kwz, kV, kiV, ilim, kturn, kthr, thmin, thmax = self.k_pitch
        kb, kib, kwy, k_ari, k_wxff = self.k_yaw
        eg = self.sp["gamma"] - g
        self.i_g = max(-0.3, min(0.3, self.i_g + eg * dtc))
        # компенсация докренивания в развороте: m_x от скорости рыскания установившегося разворота
        wy_turn = -G0 * math.sin(g) * math.cos(th) / max(V, 5.0)
        de = ks * (kwx * wx - kg * eg - kig * self.i_g) - (self.mx[2] / self.mx[3]) * wy_turn * self.l / (2 * max(V, 5.0))
        if self.emergency:
            # экстренный режим: отклонение руля высоты задается напрямую, контур скорости не работает
            dv = self.sp["elevator"]
            self.th_cmd = th
            self.dv_em = dv
        else:
            # скорость -> заданный тангаж (быстрее заданной: нос вверх), тангаж -> руль высоты
            eV = V - self.sp["V"]
            self.i_v = max(-ilim, min(ilim, self.i_v + eV * dtc))
            th_cmd = (kV * eV + kiV * self.i_v + kturn * (1.0 / max(math.cos(g), 0.5) - 1.0) + kthr * self.sp["throttle"]) * DEG
            th_cmd = max(thmin * DEG, min(thmax * DEG, th_cmd))
            if self.ramp > 0.0:
                # после экстренного режима заданный тангаж плавно переходит от фактического к контуру за 1 с
                th_cmd = (1.0 - self.ramp) * th_cmd + self.ramp * self.th_switch
                self.ramp = max(0.0, self.ramp - dtc / 1.0)
            self.th_cmd = th_cmd
            et = self.th_cmd - th
            dv = ks * (kwz * wz - kt * et)
            if self.blend > 0.0:
                dv = (1.0 - self.blend) * dv + self.blend * self.dv_em
                self.blend = max(0.0, self.blend - dtc / 1.0)
        eb = b - self.sp["beta"]
        self.i_b = max(-0.3, min(0.3, self.i_b + eb * dtc))
        de = max(-self.lim[0], min(self.lim[0], de))
        # компенсация обратного рыскания: от элеронов и от угловой скорости крена (из баланса моментов my)
        dn_ff = k_ari * de + k_wxff * wx * self.l / (2 * max(V, 5.0))
        # инерционная составляющая: момент на разгон рыскания при входе в вираж, M_y = J_y dw_y/dt
        qSl = 0.5 * self._rho(self.data.qpos[2]) * max(V, 5.0) ** 2 * self.S * self.l
        dn_ff += self.Jy * (-(G0 / max(V, 5.0)) * math.cos(g) * math.cos(th) * wx) / (qSl * self.my[4])
        dn = ks * (kb * eb + kib * self.i_b + kwy * (wy + G0 * math.sin(g) * math.cos(th) / max(V, 5.0))) + dn_ff
        cmd = np.clip(np.array([de, dv, dn]), -self.lim, self.lim)
        step = self.rate * dtc
        self.cmd = self.cmd + np.clip(cmd - self.cmd, -step, step)
        de, dv, dn = self.cmd
        self.data.ctrl[self.act] = [de, -de, -dv, dn]

    def _bumpless(self):
        """Безударный возврат в режим скорости: интеграл скорости подгоняется под текущий тангаж,
        руль высоты переходит от последнего экстренного значения к контуру за 1 с."""
        g, th, b, wx, wy, wz, V = self._attitude_now()
        kt, kwz, kV, kiV, ilim, kturn, kthr, thmin, thmax = self.k_pitch
        eV = V - self.sp["V"]
        need = max(thmin, min(thmax, th / DEG)) - kV * eV - kturn * (1.0 / max(math.cos(g), 0.5) - 1.0) - kthr * self.sp["throttle"]
        self.i_v = max(-ilim, min(ilim, need / kiV))
        self.th_switch, self.ramp = th, 1.0
        self.blend = 1.0

    def _attitude_now(self):
        pos, R, v, w = self._kin()
        g, th, _ = attitude(R)
        Vg = GOST @ (R.T @ (v - self.W))
        wg = GOST @ w
        V = float(np.linalg.norm(Vg))
        b = math.asin(max(-1.0, min(1.0, Vg[2] / max(V, 1e-6))))
        return g, th, b, wg[0], wg[1], wg[2], V

    # ---------------- шаг ----------------
    def step(self, gamma, V, beta, throttle, elevator=None):
        """
        Уставки: крен (рад), воздушная скорость (м/с), скольжение (рад), газ 0...1. Шаг 0,1 с.
        elevator: прямое отклонение руля высоты, рад (ГОСТ: + задняя кромка вниз, пикирование),
        экстренный режим; None - обычный режим, руль высоты держит заданную скорость.
        """
        g_l, b_l, v_lo, v_hi = self.sp_lim
        em = elevator is not None
        was = self.emergency
        self.sp = dict(gamma=float(np.clip(gamma, -g_l, g_l)), V=float(np.clip(V, v_lo, v_hi)),
                       beta=float(np.clip(beta, -b_l, b_l)), throttle=float(np.clip(throttle, 0.0, 1.0)),
                       elevator=float(np.clip(elevator, -self.lim[1], self.lim[1])) if em else None)
        self.emergency = em
        if was and not em:
            self._bumpless()
        for k in range(self.n_sub):
            self._substep(k)
            if self.crashed:
                break
        self.model.geom_rgba[self.prop_geom, 3] = 0.35 if self.level > 0.05 else 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def _substep(self, k):
        m, d = self.model, self.data
        if k % self.n_ctrl == 0:
            self._pid(self.dt * self.n_ctrl)
        pos, R, v, w = self._kin()
        rho = self._rho(pos[2])
        Vg = GOST @ (R.T @ (v - self.W))
        F, Mo, info = self.forces(Vg, GOST @ w, self._deflections(), rho)
        cmd = self.sp["throttle"] if self.sp["throttle"] >= self.dead else 0.0
        if self.E <= 0.0:
            cmd = 0.0
        if cmd > 0 and not self.motor_on:
            self.motor_on, self.on_time = True, 0.0
        if cmd == 0 and self.motor_on and self.level < 0.02:
            self.motor_on = False
        if self.motor_on:
            self.on_time += self.dt
        target = cmd if (self.motor_on and self.on_time >= self.unfold) else 0.0
        self.level += (target - self.level) * min(1.0, self.dt / self.tau)
        self.P, self.N = self.thrust(self.level, info["V"], rho)
        self.E -= (self.N + self.P_av) * self.dt
        self.E_motor += self.N * self.dt
        F = F + np.array([self.P, 0.0, 0.0])
        F_w = R @ (GOST.T @ F)
        d.xfrc_applied[self.bid, 0:3] = F_w
        d.xfrc_applied[self.bid, 3:6] = R @ (GOST.T @ Mo) + np.cross(R @ self.lever, F_w)
        info["ny"] = F[1] / (self._mass() * G0)
        self.aero = info
        self._move_obstacles(self.t + self.dt)
        mujoco.mj_step(m, d)
        self.t += self.dt
        if d.warning[self.bad].number > self.warn0:
            self.crashed, self.reason = True, "неустойчивость интегрирования"
        elif d.ncon > 0:
            c = d.contact[0]
            other = c.geom2 if m.geom_bodyid[c.geom1] in self._glider_bodies() else c.geom1
            self.crashed = True
            if other == self.ground:
                self.reason = "земля"
            elif other == self.sudden_geom:
                self.reason = "внезапное препятствие"
            elif other in self.mover_geoms:
                self.reason = "движущееся препятствие " + self.mover_geoms[other]
            else:
                self.reason = "препятствие"
        elif info["V"] > self.kV * self.V_NE:
            self.crashed, self.reason = True, "разрушение: превышение скорости"
        elif abs(info["ny"]) > self.kn * self.n_max:
            self.crashed, self.reason = True, "разрушение: перегрузка"

    def _glider_bodies(self):
        if not hasattr(self, "_gb"):
            m = self.model
            self._gb = {b for b in range(m.nbody) if m.body_rootid[b] == self.bid}
        return self._gb

    # ---------------- наблюдения ----------------
    def state(self):
        pos, R, v, w = self._kin()
        g, th, hd = attitude(R)
        ai = self.aero
        goal = self.data.mocap_pos[self.goal_id].copy()
        h_dir = np.array([math.sin(hd), math.cos(hd)])
        right = np.array([h_dir[1], -h_dir[0]])
        alpha_max = 11.0 * DEG
        return dict(
            t=self.t, pos=pos, vel=v, V=ai["V"], alpha=ai["alpha"], beta=ai["beta"], gamma=g, theta=th, heading=hd,
            omega=GOST @ w, ny=ai.get("ny", 1.0), H=pos[2], cya=ai["cya"], cxa=ai["cxa"],
            wind=self.W.copy(), headwind=-float(self.W[:2] @ h_dir), crosswind=-float(self.W[:2] @ right),
            goal=goal, goal_dist=float(np.linalg.norm(goal - pos)), battery=max(0.0, self.E / self.E_total),
            motor_on=self.level > 0.02, thrust=self.P, power=self.N, E_motor=self.E_motor,
            deflections=np.array(self._deflections()), setpoints=dict(self.sp), theta_cmd=self.th_cmd, emergency=self.emergency,
            stalled=ai["alpha"] > alpha_max + 2 * DEG, crashed=self.crashed, reason=self.reason)

    def wind(self, noise=0.0):
        """Вектор ветра в мировой системе и в осях курса: встречная и боковая (справа +) составляющие."""
        W = self.W + noise * self.rng.standard_normal(3) * np.array([1, 1, 0])
        s = self.state()
        return dict(world=W, headwind=s["headwind"], crosswind=s["crosswind"])

    def obstacles(self):
        """Препятствия: положение, размеры, скорость (для сети-учителя и проверок)."""
        m, d = self.model, self.data
        out = []
        for gid in self.static:
            out.append(dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid), kind="статичное",
                            type=int(m.geom_type[gid]), pos=d.geom_xpos[gid].copy(), size=m.geom_size[gid].copy(), vel=np.zeros(3)))
        for mv in self.movers:
            p, vel = self.mover_state(mv, self.t)
            out.append(dict(name=mv["name"], kind=MOVE_KIND[mv["kind"]], type=int(m.geom_type[mv["geom"]]), pos=p,
                            size=m.geom_size[mv["geom"]].copy(), vel=vel))
        if self.sudden_active:
            out.append(dict(name="sudden_obstacle", kind="внезапное", type=int(m.geom_type[self.sudden_geom]),
                            pos=d.mocap_pos[self.sudden_id].copy(), size=m.geom_size[self.sudden_geom].copy(), vel=np.zeros(3)))
        return out

    # ---------------- сцены ----------------
    def spawn_obstacle(self, distance=50.0, lateral=0.0, dz=0.0):
        """Внезапное препятствие (шар радиусом 6 м) на distance м по курсу, lateral вправо, dz вверх."""
        pos, R, v, w = self._kin()
        h = v[:2] / (np.linalg.norm(v[:2]) + 1e-9)
        right = np.array([h[1], -h[0]])
        p = np.array([pos[0] + h[0] * distance + right[0] * lateral, pos[1] + h[1] * distance + right[1] * lateral, pos[2] + dz])
        self.data.mocap_pos[self.sudden_id] = p
        self.sudden_active = True
        mujoco.mj_forward(self.model, self.data)
        return p

    def hide_obstacle(self):
        self.data.mocap_pos[self.sudden_id] = [0.0, 0.0, -2000.0]
        self.sudden_active = False

    def default_scene(self):
        """Сцена по умолчанию из motoglider.xml."""
        m = self.model
        self.scene_pos = self._base["pos"].copy()
        self.data.mocap_pos[self.static_mocap] = self.scene_pos
        m.geom_size[self.static] = self._base["size"]
        m.geom_rbound[self.static] = self._base["rbound"]
        m.geom_aabb[self.static] = self._base["aabb"]
        m.bvh_aabb[m.body_bvhadr[self.static_body]] = self._base["bvh"]
        for mv, (p0, ph, v0) in zip(self.movers, self._base["movers"]):
            mv["p0"], mv["phase0"], mv["v0"] = p0.copy(), ph, v0.copy()

    def randomize_scene(self, rng=None, start=(0.0, 0.0, 130.0), dist=(800.0, 2500.0), corridor=200.0, scale=(0.7, 1.4)):
        """Случайная сцена: цель в случайном направлении, препятствия случайного размера в коридоре
        вдоль маршрута, новые точки привязи и фазы движущихся препятствий. Возвращает старт, цель, курс."""
        rng = rng or self.rng
        m = self.model
        start = np.asarray(start, dtype=float)
        L = rng.uniform(*dist)
        brg = rng.uniform(0.0, 2 * math.pi)
        u = np.array([math.sin(brg), math.cos(brg)])
        nrm = np.array([-u[1], u[0]])
        goal = np.array([start[0] + u[0] * L, start[1] + u[1] * L, rng.uniform(60.0, 200.0)])
        for k, gid in enumerate(self.static):
            s = rng.uniform(*scale)
            size = self._base["size"][k] * s
            m.geom_size[gid] = size
            m.geom_rbound[gid] = self._base["rbound"][k] * s
            m.geom_aabb[gid] = np.r_[self._base["aabb"][k][:3] * s, self._base["aabb"][k][3:] * s]
            m.bvh_aabb[m.body_bvhadr[self.static_body[k]]] = np.r_[self._base["bvh"][k][:3] * s, self._base["bvh"][k][3:] * s]
            xy = start[:2] + u * rng.uniform(0.15, 0.9) * L + nrm * rng.uniform(-corridor, corridor)
            t = m.geom_type[gid]
            if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                z = size[1]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                z = size[2]
            elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                z = size[2]
            else:
                z = rng.uniform(60.0, 160.0)
            self.scene_pos[k] = [xy[0], xy[1], z]
        for mv in self.movers:
            xy = start[:2] + u * rng.uniform(0.2, 0.95) * L + nrm * rng.uniform(-corridor, corridor)
            mv["p0"] = np.array([xy[0], xy[1], rng.uniform(70.0, 160.0)])
            mv["phase0"] = rng.uniform(0.0, 2 * math.pi)
            if mv["kind"] == 2:
                a = rng.uniform(0.0, 2 * math.pi)
                mv["v0"] = rng.uniform(1.0, 4.0) * np.array([math.cos(a), math.sin(a), 0.0])
        heading = (math.degrees(brg) + rng.uniform(-30.0, 30.0)) % 360.0
        self.data.mocap_pos[self.static_mocap] = self.scene_pos
        mujoco.mj_forward(m, self.data)
        return start, goal, heading

    def rays(self, max_range=300.0):
        """75 дальностей по лучам из носа (15 по горизонтали, 5 по вертикали), доля от max_range."""
        pos, R, _, _ = self._kin()
        nose = pos + R @ np.array([0.65, 0.0, 0.0])
        dirs = np.ascontiguousarray((self.ray_dirs @ R.T).reshape(-1))
        n = len(self.ray_dirs)
        gid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        try:
            mujoco.mj_multiRay(self.model, self.data, nose, dirs, self.ray_group, 1, self.bid, gid, dist, None, n, max_range)
        except TypeError:
            mujoco.mj_multiRay(self.model, self.data, nose, dirs, self.ray_group, 1, self.bid, gid, dist, n, max_range)
        dist = np.where((dist < 0) | (dist > max_range), max_range, dist)
        return (dist / max_range).astype(np.float32)

    def depth_true(self, camera="cam_left", renderer=None):
        rd = renderer or self.renderer
        rd.update_scene(self.data, camera=camera, scene_option=self.scene_opt)
        rd.enable_depth_rendering()
        z = rd.render().copy()
        rd.disable_depth_rendering()
        return z

    def depth(self, renderer=None):
        """Карта дальности левой камеры с ошибками стереопары (inf - нет данных)."""
        D = self.depth_true(renderer=renderer).astype(float)
        D = np.where(D >= 0.99 * self.zfar, np.inf, D)
        return self.stereo_errors(D, self.aero.get("ny", 1.0))

    def stereo_errors(self, D, ny):
        """Ошибки стереопары: шум диспаратности и смещение от изгиба консолей."""
        dpsi = self.resid * ((self.flex * (ny - 1.0) + self.mis) * DEG + 0.02 * DEG * self.rng.standard_normal())
        disp = self.f_px * self.B / D + self.f_px * dpsi + self.sig_d * self.rng.standard_normal(D.shape)
        out = np.where(disp > self.d_min, self.f_px * self.B / np.maximum(disp, 1e-6), np.inf)
        return np.where(out > self.D_max, np.inf, out).astype(np.float32)

    def sigma_D(self, D):
        """СКО дальности стереопары: D^2 sigma_d / (f B)."""
        return D * D * self.sig_d / (self.f_px * self.B)

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None


def polar(sim, mass=None, rho=None):
    """Характеристики поляры без рулей: Kmax, скорость Kmax, сваливание, минимальное снижение."""
    mass = mass or sim._mass()
    rho = rho or sim.rho0
    al = np.linspace(-0.05, 11.0 * DEG, 3000)
    cy, cx = [], []
    for a in al:
        F, _, info = sim.forces((math.cos(a), -math.sin(a), 0.0), (0, 0, 0), (0, 0, 0), 1.0)
        cy.append(info["cya"])
        cx.append(info["cxa"])
    cy, cx = np.array(cy), np.array(cx)
    ok = cy > 0.05
    V = np.sqrt(2 * mass * G0 / (rho * sim.S * np.maximum(cy, 1e-3)))
    K = np.where(ok, cy / cx, 0.0)
    i = int(np.argmax(K))
    sink = np.where(ok, V / np.maximum(K, 1e-6), 1e9)
    j = int(np.argmin(sink))
    return dict(K_max=float(K[i]), V_Kmax=float(V[i]), sink_min=float(sink[j]), V_sink=float(V[j]),
                cya_max=float(cy.max()), V_stall=math.sqrt(2 * mass * G0 / (rho * sim.S * cy.max())))


def glide_K(sim, V, mass=None, rho=None):
    """Качество и вертикальная скорость установившегося планирования на скорости V."""
    mass = mass or sim._mass()
    rho = rho or sim.rho0
    cy_req = 2 * mass * G0 / (rho * sim.S * V * V)
    lo, hi = -0.1, 11.0 * DEG
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sim.forces((math.cos(mid), -math.sin(mid), 0.0), (0, 0, 0), (0, 0, 0), 1.0)[2]["cya"] < cy_req:
            lo = mid
        else:
            hi = mid
        info = sim.forces((math.cos(mid), -math.sin(mid), 0.0), (0, 0, 0), (0, 0, 0), 1.0)[2]
    return info["cya"] / info["cxa"], V * info["cxa"] / info["cya"]
