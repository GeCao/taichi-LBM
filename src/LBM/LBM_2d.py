import taichi as ti
from .LBM_abstract import LBMAbstract
from typing import List, TypeVar, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBM_2D(LBMAbstract):
    def __init__(self, lbm_manager: "LBMManager"):
        super(LBM_2D, self).__init__(lbm_manager=lbm_manager)
        self._e = ti.field(dtype=ti.i32, shape=(self._Q, 2))
        self._w = ti.field(dtype=ti.f32, shape=(self._Q, ))

        np_e = np.array([[0, 0],
                         [1, 0], [0, 1], [-1, 0], [0, -1],
                         [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float)
        self._e.from_numpy(np_e)

        np_w = np.array([4.0 / 9.0,
                         1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                         1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float)
        self._w.from_numpy(np_w)

    @ti.func
    def initialization(self):
        Vmax = self._Vmax
        resolution = self._resolution
        visc = self._visc
        time = self.m_time[None]
        Ma = self._Ma

        # params for case 0:
        lambda_alpha = self._case_parameters[0]['lambda_alpha']
        K_x, K_y = 2 * math.pi / lambda_alpha / resolution[0], 2 * math.pi / lambda_alpha / resolution[1]
        K = ti.sqrt(K_x * K_x + K_y * K_y)

        # params for case 1:
        KBC_sigma = self._case_parameters[1]['sigma']
        KBC_kappa = self._case_parameters[1]['kappa']

        for i, j in self.m_rho:
            if self.m_cell_type[i, j] == 0:
                if self.m_case[None] == 0:
                    self.m_velocity[i, j, 0] = -Vmax * K * ti.exp(-visc * K * K * time) * \
                                           ti.sin(K_y * j) * ti.cos(K_x * i)
                    self.m_velocity[i, j, 1] = Vmax * K_x / K * ti.exp(-visc * K * K * time) * \
                                           ti.sin(K_x * i) * ti.cos(K_y * j)
                    self.m_rho[i, j] = 1 - Ma * Ma / 2.0 / K / K * \
                                   (K_y * K_y * ti.cos(2.0 * K_x * i) + K_x * K_x * ti.cos(2.0 * K_y * j))
                elif self.m_case[None] == 1:
                    if self.m_cell_type[i, j] == 0:
                        self.m_velocity[i, j, 1] = KBC_sigma * Vmax * \
                                                   ti.sin(2.0 * math.pi * (1.0 * i / resolution[0] + 0.25))
                        if j <= resolution[1] / 2.0:
                            self.m_velocity[i, j, 0] = Vmax * ti.tanh(KBC_kappa * (1.0 * j / resolution[1] - 0.25))
                        else:
                            self.m_velocity[i, j, 0] = Vmax * ti.tanh(KBC_kappa * (-1.0 * j / resolution[1] + 0.75))
                        self.m_rho[i, j] = 0.265
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        for I in ti.grouped(self.m_rho):
            u_tackle[0], u_tackle[1] = self.m_velocity[I, 0], self.m_velocity[I, 1]
            self.cal_feq(u_tackle, I)
        for I in ti.grouped(self.m_f):
            self.m_f[I] = self.m_feq[I]


    @ti.func
    def run_step(self):
        self.streaming()
        self.macro_variable_process()
        self.cal_forces()
        self.rebounce_f()
        self.collision()

        self.m_time[None] += self._dt
        self.compute_image()
        # self.error_check()

    @ti.func
    def streaming(self):
        wid, hei = self.m_rho.shape
        for i, j in self.m_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0
            j_down = j - 1 if j > 0 else hei - 1
            j_up = j + 1 if j < hei - 1 else 0

            self.m_feq[i, j, 0] = self.m_f[i, j, 0]

            self.m_feq[i_right, j, 1] = self.m_f[i, j, 1]
            self.m_feq[i, j_up, 2] = self.m_f[i, j, 2]
            self.m_feq[i_left, j, 3] = self.m_f[i, j, 3]
            self.m_feq[i, j_down, 4] = self.m_f[i, j, 4]

            self.m_feq[i_right, j_up, 5] = self.m_f[i, j, 5]
            self.m_feq[i_left, j_up, 6] = self.m_f[i, j, 6]
            self.m_feq[i_left, j_down, 7] = self.m_f[i, j, 7]
            self.m_feq[i_right, j_down, 8] = self.m_f[i, j, 8]

        for I in ti.grouped(self.m_f):
            self.m_f[I] = self.m_feq[I]

    @ti.func
    def rebounce_f(self):
        for I in ti.grouped(self.m_cell_type):
            if self.m_cell_type[I] == 1:
                self.m_f[I, 1], self.m_f[I, 3] = self.m_f[I, 3], self.m_f[I, 1]
                self.m_f[I, 2], self.m_f[I, 4] = self.m_f[I, 4], self.m_f[I, 2]
                self.m_f[I, 5], self.m_f[I, 7] = self.m_f[I, 7], self.m_f[I, 5]
                self.m_f[I, 6], self.m_f[I, 8] = self.m_f[I, 8], self.m_f[I, 6]

    @ti.func
    def cal_feq(self, vel, I):
        for k in range(self._Q):
            self.m_feq[I, k] = self.m_rho[I] * self._w[k] * \
                               (2.0 - ti.sqrt(1.0 + 3.0 * vel[0] * vel[0])) * \
                               (2.0 - ti.sqrt(1.0 + 3.0 * vel[1] * vel[1])) * \
                               ti.pow((2.0 * vel[0] + ti.sqrt(1.0 + 3.0 * vel[0] * vel[0])) / (1.0 - vel[0]), self._e[k, 0]) * \
                               ti.pow((2.0 * vel[1] + ti.sqrt(1.0 + 3.0 * vel[1] * vel[1])) / (1.0 - vel[1]), self._e[k, 1])

    @ti.func
    def collision(self):
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        tau = self._tau
        for I, in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                u_tackle[0] = self.m_velocity[I, 0] + tau * (self.m_force[I, 0]) / self.m_rho[I]
                u_tackle[1] = self.m_velocity[I, 1] + tau * (self.m_force[I, 1]) / self.m_rho[I]
                self.cal_feq(u_tackle, I)
                for k in range(self._Q):
                    self.m_f[I, k] = (1.0 - 1.0 / tau) * self.m_f[I, k] + self.m_feq[I, k] / tau

    @ti.func
    def macro_variable_process(self):
        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                self.m_rho[I] = 0.0
                for rr in ti.static(range(self._dim)):
                    self.m_velocity[I, rr] = 0.0

        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                for k in ti.static(range(self._Q)):
                    self.m_rho[I] += self.m_f[I, k]
                    for rr in ti.static(range(self._dim)):
                        self.m_velocity[I, rr] += self.m_f[I, k] * self._e[k, rr]

        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                for rr in ti.static(range(self._dim)):
                    self.m_velocity[I, rr] *= (self._c / self.m_rho[I])

    @ti.func
    def cal_forces(self):
        pass

    @ti.func
    def compute_vort(self):
        wid, hei = self.m_rho.shape
        for i, j in self.m_vort:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0
            j_down = j - 1 if j > 0 else hei - 1
            j_up = j + 1 if j < hei - 1 else 0
            self.m_vort[i, j] = (self.m_velocity[i, j_up, 0] - self.m_velocity[i, j_down, 0]) / (2.0 * self._dx) - \
                                (self.m_velocity[i_right, j, 1] - self.m_velocity[i_left, j, 1]) / (2.0 * self._dx)

        for i, j in self.m_vort:
            if self.m_vort[i, j] > 0:
                self.m_image[i, j, 0], self.m_image[i, j, 1], self.m_image[i, j, 2] = self.m_vort[i, j] * 255.0, 0, 0
            elif self.m_vort[i, j] < 0:
                self.m_image[i, j, 0], self.m_image[i, j, 1], self.m_image[i, j, 2] = 0, 0, -self.m_vort[i, j] * 255.0
            else:
                self.m_image[i, j, 0], self.m_image[i, j, 1], self.m_image[i, j, 2] = 0, 0, 0

    @ti.func
    def compute_image(self):
        image_is_vort = self._case_parameters[self._lbm_manager._case]['image'] == 'vort'
        image_is_rho = self._case_parameters[self._lbm_manager._case]['image'] == 'rho'
        image_is_velocity = self._case_parameters[self._lbm_manager._case]['image'] == 'velocity'
        if image_is_vort:
            self.compute_vort()
        for I in ti.grouped(self.m_rho):
            if image_is_vort:
                if self.m_vort[I] > 0:
                    self.m_image[I, 0], self.m_image[I, 1], self.m_image[I, 2] = self.m_vort[I] * 255.0, 0, 0
                elif self.m_vort[I] < 0:
                    self.m_image[I, 0], self.m_image[I, 1], self.m_image[I, 2] = 0, 0, -self.m_vort[I] * 255.0
                else:
                    self.m_image[I, 0], self.m_image[I, 1], self.m_image[I, 2] = 0, 0, 0
            elif image_is_rho:
                self.m_image[I, 0], self.m_image[I, 1], self.m_image[I, 2] = \
                    self.m_rho[I] * 1.0, self.m_rho[I] * 1.0, self.m_rho[I] * 1.0
            elif image_is_velocity:
                self.m_image[I, 0], self.m_image[I, 1], self.m_image[I, 2] = self.m_velocity[I, 0] * 32.0, \
                                                                             self.m_velocity[I, 1] * 32.0, 0

    @ti.func
    def error_check(self):
        pass

    @ti.kernel
    def kill(self):
        pass
