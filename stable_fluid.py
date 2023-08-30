# reference implementation:
# https://github.com/taichi-dev/tai-objc-runtime/blob/master/stable_fluid.py

# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation

import taichi as ti
import numpy as np
import colorsys

ti.init(arch=ti.gpu)

# resolution constants
SIM_RES = 128
RENDER_RES_X = 1024
RENDER_RES_Y = 768
aspect_ratio = float(RENDER_RES_X) / float(RENDER_RES_Y)
SIM_RES_Y = SIM_RES
SIM_RES_X = int(np.ceil(SIM_RES_Y * aspect_ratio))

# sim constants
max_fps = 60
dt = 1.0 / max_fps
# assuming grid size dx = 1
force_radius = 0.1 / 100
inv_force_radius = 1.0 / force_radius
dye_radius = 0.1 / 100
inv_dye_radius = 1.0 / dye_radius
f_strength = 5000
curl_strength = 30.0

# numerical solver constants
p_jacobi_iters = 20
p_jacobi_warm_starting = 0.8 # 0 to 1

# index naming convention
# i, j for index, scales from 0 to SIM/RENDER_RES_X/Y - 1
# u, v for texture space coordinates, scale from 0 to 1
# all quantities are stored at the grid centers

# handy helper functions
@ti.func
def sample_clamp_to_edge(qf, i, j, res_x, res_y):
    i, j = int(i), int(j)
    # clamp to edge
    i = max(0, min(res_x - 1, i))
    j = max(0, min(res_y - 1, j))
    return qf[i, j]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def bilerp(qf, i, j, res_x, res_y):
    ii, ij = int(ti.floor(i)), int(ti.floor(j)) # integer part
    fi, fj = i - ii, j - ij # fraction part
    a = sample_clamp_to_edge(qf, ii, ij, res_x, res_y)
    b = sample_clamp_to_edge(qf, ii + 1, ij, res_x, res_y)
    c = sample_clamp_to_edge(qf, ii, ij + 1, res_x, res_y)
    d = sample_clamp_to_edge(qf, ii + 1, ij + 1, res_x, res_y)
    return lerp(lerp(a, b, fi), lerp(c, d, fi), fj)

class TexPair:
    def __init__(self, cur, nxt):
        assert isinstance(cur, Texture)
        assert isinstance(nxt, Texture)
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

@ti.data_oriented
class Texture(object):
    def __init__(self, f, res_x, res_y):
        self.field = f
        self.res_x = res_x
        self.res_y = res_y
        self.texel_size_x = 1.0 / res_x
        self.texel_size_y = 1.0 / res_y
        self.texel_size = self.texel_size_y

    @staticmethod
    def Scalar(res_x, res_y):
        f = ti.field(ti.f32, shape=(res_x, res_y))
        return Texture(f, res_x, res_y)

    @staticmethod
    def Vector(dim, res_x, res_y):
        f = ti.Vector.field(dim, dtype=ti.f32, shape=(res_x, res_y))
        return Texture(f, res_x, res_y)

    @ti.func
    def sample_sep(self, u, v):
        i = u * self.res_x - 0.5
        j = v * self.res_y - 0.5
        return bilerp(self.field, i, j, self.res_x, self.res_y)

    @ti.func
    def sample(self, uv):
        return self.sample_sep(uv[0], uv[1])

    @ti.func
    def normalize(self, ij):
        u = ij[0] * self.texel_size_x
        v = ij[1] * self.texel_size_y
        return ti.Vector([u, v])


# taichi fields
# simulation quantities
_velocities = Texture.Vector(2, SIM_RES_X, SIM_RES_Y)
_new_velocities = Texture.Vector(2, SIM_RES_X, SIM_RES_Y)
_velocity_divs = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_velocity_curls = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_pressures = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_new_pressures = Texture.Scalar(SIM_RES_X, SIM_RES_Y)

# visualization quantities
_dye_buffer = Texture.Vector(3, RENDER_RES_X, RENDER_RES_Y)
_new_dye_buffer = Texture.Vector(3, RENDER_RES_X, RENDER_RES_Y)

pixels = Texture.Vector(3, RENDER_RES_X, RENDER_RES_Y)

# double buffered quantities
velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

# simulation kernels
@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
           dissipation: float):
    for i, j in qf.field:
        uv = qf.normalize(ti.Vector([i, j]) + 0.5)
        vel = vf.sample(uv)
        vel[0] *= vf.texel_size_x
        vel[1] *= vf.texel_size_y # transfer to uv space
        prev_uv = uv - dt * vel # backtracing, RK-1
        prev_q = qf.sample(prev_uv) 
        decay = 1.0 + dissipation * dt
        new_qf.field[i, j] = prev_q / decay

@ti.kernel
def splat_velocity(vf: ti.template(), omx: float, omy: float, fx: float, fy: float):
    for i, j in vf.field:
        u, v = vf.normalize(ti.Vector([i, j]) + 0.5)
        dx, dy = (u - omx) * aspect_ratio, (v - omy)
        d2 = dx * dx + dy * dy
        momentum = ti.exp(-d2 * inv_force_radius) * ti.Vector([fx, fy]) * f_strength
        vf.field[i, j] += momentum

@ti.kernel
def splat_dye(dye: ti.template(), omx: float, omy: float, r: float, g: float, b: float):
    for i, j in dye.field:
        u, v = dye.normalize(ti.Vector([i, j]) + 0.5)
        dx, dy = (u - omx) * aspect_ratio, (v - omy)
        d2 = dx * dx + dy * dy
        color = ti.exp(-d2 * inv_dye_radius) * ti.Vector([r, g, b])
        dye.field[i, j] += color

@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in _velocity_curls.field:
        res_x = vf.res_x
        res_y = vf.res_y
        vl = sample_clamp_to_edge(vf.field, i - 1, j, res_x, res_y)[1]
        vr = sample_clamp_to_edge(vf.field, i + 1, j, res_x, res_y)[1]
        vb = sample_clamp_to_edge(vf.field, i, j - 1, res_x, res_y)[0]
        vt = sample_clamp_to_edge(vf.field, i, j + 1, res_x, res_y)[0]
        vort = vr - vl - vt + vb
        _velocity_curls.field[i, j] = 0.5 * vort

@ti.kernel
def vorticity_confinement(vf: ti.template()):
    for i, j in vf.field:
        res_x = vf.res_x
        res_y = vf.res_y
        vl = sample_clamp_to_edge(_velocity_curls.field, i - 1, j, res_x, res_y)
        vr = sample_clamp_to_edge(_velocity_curls.field, i + 1, j, res_x, res_y)
        vb = sample_clamp_to_edge(_velocity_curls.field, i, j - 1, res_x, res_y)
        vt = sample_clamp_to_edge(_velocity_curls.field, i, j + 1, res_x, res_y)
        vc = sample_clamp_to_edge(_velocity_curls.field, i, j, res_x, res_y)

        force = 0.5 * ti.Vector([abs(vt) - abs(vb),
                                 abs(vl) - abs(vr)]).normalized(1e-4)
        force *= curl_strength * vc
        vel = vf.field[i, j]
        vf.field[i, j] = min(max(vel + force * dt, -1e3), 1e3)

@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf.field:
        res_x = vf.res_x
        res_y = vf.res_y
        vl = sample_clamp_to_edge(vf.field, i - 1, j, res_x, res_y)[0]
        vr = sample_clamp_to_edge(vf.field, i + 1, j, res_x, res_y)[0]
        vb = sample_clamp_to_edge(vf.field, i, j - 1, res_x, res_y)[1]
        vt = sample_clamp_to_edge(vf.field, i, j + 1, res_x, res_y)[1]
        vc = sample_clamp_to_edge(vf.field, i, j, res_x, res_y)
        if i == 0:
            vl = -vc[0]
        if i == vf.res_x - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == vf.res_y - 1:
            vt = -vc[1]
        _velocity_divs.field[i, j] = 0.5 * (vr - vl + vt - vb)

@ti.kernel
def warm_start_pressure(pf: ti.template()):
    for i, j in pf.field:
        pf.field[i, j] *= p_jacobi_warm_starting

@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf.field:
        res_x = pf.res_x
        res_y = pf.res_y
        pl = sample_clamp_to_edge(pf.field, i - 1, j, res_x, res_y)
        pr = sample_clamp_to_edge(pf.field, i + 1, j, res_x, res_y)
        pb = sample_clamp_to_edge(pf.field, i, j - 1, res_x, res_y)
        pt = sample_clamp_to_edge(pf.field, i, j + 1, res_x, res_y)
        div = _velocity_divs.field[i, j]
        new_pf.field[i, j] = (pl + pr + pb + pt - div) * 0.25

@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf.field:
        res_x = vf.res_x
        res_y = vf.res_y
        pl = sample_clamp_to_edge(pf.field, i - 1, j, res_x, res_y)
        pr = sample_clamp_to_edge(pf.field, i + 1, j, res_x, res_y)
        pb = sample_clamp_to_edge(pf.field, i, j - 1, res_x, res_y)
        pt = sample_clamp_to_edge(pf.field, i, j + 1, res_x, res_y)
        vel = sample_clamp_to_edge(vf.field, i, j, res_x, res_y)
        vel -= 0.5 * ti.Vector([pr - pl, pt - pb])
        vf.field[i, j] = vel

# simulation control flow
def reset():
    velocities_pair.cur.field.fill(ti.Vector([0, 0]))
    pressures_pair.cur.field.fill(0.0)
    dyes_pair.cur.field.fill(ti.Vector([0, 0, 0]))
    pixels.field.fill(ti.Vector([0, 0, 0]))

def apply_impulse(mouse_data):

    normed_mxy = mouse_data[3:5]
    delta_mxy = mouse_data[1:3]
    color = mouse_data[5:8]

    splat_velocity(velocities_pair.cur, float(normed_mxy[0]), float(normed_mxy[1]),
                     float(delta_mxy[0]), float(delta_mxy[1]))
    
    splat_dye(dyes_pair.cur, float(normed_mxy[0]), float(normed_mxy[1]), float(color[0]),
                float(color[1]), float(color[2]))

def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt, 0.0)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, 0.8)
    velocities_pair.swap()
    dyes_pair.swap()

    if mouse_data[0]:
        apply_impulse(mouse_data)

    if curl_strength:
        vorticity(velocities_pair.cur)
        vorticity_confinement(velocities_pair.cur)

    divergence(velocities_pair.cur)
    warm_start_pressure(pressures_pair.cur)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

# rendering kernels
@ti.kernel
def fill_color_v3(dye: ti.template()):
    for i, j in pixels.field:
        uv = dye.normalize(ti.Vector([i, j]) + 0.5)
        v = dye.sample(uv)
        c = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])

        pixels.field[i, j] = c

# rendering control flow
def render():
    fill_color_v3(dyes_pair.cur)

# random color gen
def generate_color():
    c = np.array(colorsys.hsv_to_rgb(np.random.random(), 1.0, 1.0))
    c *= 1.0
    return c

# mouse events
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None
        self.mouse_ticks = 0
        self.change_color = False

    def __call__(self, window):
        # [0]: whether mouse is moved
        # [1:3]: normalized delta direction
        # [3:5]: current mouse xy
        # [5:8]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if window.is_pressed(ti.ui.LMB):
            mxy = np.array(window.get_cursor_pos(), dtype=np.float32)

            # change dye color every 10 mouse events
            if self.change_color and self.mouse_ticks > 9:
                self.mouse_ticks = 0
                self.prev_color = generate_color()

            if self.prev_mouse is None: # mouse pressed
                self.mouse_ticks = 0
                self.prev_mouse = mxy
                self.prev_color = generate_color()
            else: # mouse moving
                self.mouse_ticks += 1
                delta_mxy = mxy - self.prev_mouse
                if np.linalg.norm(delta_mxy) > 1e-4:
                    mouse_data[0] = 1
                    mouse_data[1], mouse_data[2] = delta_mxy[0], delta_mxy[1]
                    mouse_data[3], mouse_data[4] = mxy[0], mxy[1]
                    mouse_data[5:8] = self.prev_color
                self.prev_mouse = mxy
                
        else:
            mouse_data[0] = 0
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

def main():
    window = ti.ui.Window('Fancy Stable Fluids', (RENDER_RES_X, RENDER_RES_Y), vsync=True)
    canvas = window.get_canvas()
    md_gen = MouseDataGen()
    paused = False

    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == 'r':
                paused = False
                reset()
            elif e.key == 'p':
                paused = not paused
            elif e.key == 'c':
                md_gen.change_color = not md_gen.change_color 

        if not paused:
            mouse_data = md_gen(window)
            step(mouse_data)
            render()
            
        canvas.set_image(pixels.field)
        window.show()

if __name__ == '__main__':
    main()