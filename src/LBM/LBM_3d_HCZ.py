import taichi as ti
from .LBM_3d import LBM_3D
from typing import List, TypeVar, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBM_3D_HCZ(LBM_3D):
    def __init__(self, lbm_manager: "LBMManager"):
        super(LBM_3D_HCZ, self).__init__(lbm_manager=lbm_manager)
        # new_fields
        self.m_psi = ti.field(ti.f32, shape=self._resolution)
        self.m_g = ti.field(ti.f32, shape=(*self._resolution, self._Q))
        self.m_fai = ti.field(ti.f32, shape=(*self._resolution,))
        self.m_prho = ti.field(ti.f32, shape=(*self._resolution,))
        self.m_dfai = ti.field(ti.f32, shape=(*self._resolution, self._dim))
        self.m_dprho = ti.field(ti.f32, shape=(*self._resolution, self._dim))
        self.m_density = ti.field(ti.f32, shape=(*self._resolution,))
        self.m_laplacian_rho = ti.field(ti.f32, shape=(*self._resolution,))

        # Ghost area
        wid, hei, dep = self._resolution
        self.m_ghost_left = ti.field(ti.f32, shape=(hei, dep))  # -x
        self.m_ghost_right = ti.field(ti.f32, shape=(hei, dep))  # +x
        self.m_ghost_front = ti.field(ti.f32, shape=(wid, dep))  # -y
        self.m_ghost_behind = ti.field(ti.f32, shape=(wid, dep))  # +y
        self.m_ghost_down = ti.field(ti.f32, shape=(wid, hei))  # -z
        self.m_ghost_up = ti.field(ti.f32, shape=(wid, hei))  # +z

        # re-parameter:
        self._rho_liquid = self._case_parameters[self._lbm_manager._case]['rho_liquid']
        self._rho_gas = self._case_parameters[self._lbm_manager._case]['rho_gas']
        self._tau_f = self._case_parameters[self._lbm_manager._case]['tau_f']
        self._tau_g = self._case_parameters[self._lbm_manager._case]['tau_g']
        self._density_liquid = self._case_parameters[self._lbm_manager._case]['density_liquid']
        self._density_gas = self._case_parameters[self._lbm_manager._case]['density_gas']
        self._psi_max = self._case_parameters[self._lbm_manager._case]['psi_max']
        self._psi_min = self._case_parameters[self._lbm_manager._case]['psi_min']
        self._kappa = self._case_parameters[self._lbm_manager._case]['kappa']
        self._theta_A = self._case_parameters[self._lbm_manager._case]['theta_A']
        self._theta_R = self._case_parameters[self._lbm_manager._case]['theta_R']
        self._wall = self._case_parameters[self._lbm_manager._case]['wall']

        self._visc = self._cs2 * (self._tau_f - 0.5)
        self._Vmax = self._Re * self._visc / self._Lmax
        self._Ma = self._visc / self._cs2

        self._b = 4.0
        self._a = 12.0 * self._cs2
        self._RT = self._cs2

    @ti.func
    def initialization(self):
        b = self._b
        a = self._a
        RT = self._RT
        for i, j, r in self.m_rho:
            if ti.abs(i - self._resolution[0] * 0.5) < 0.25 * self._resolution[0] and \
                    ti.abs(j - self._resolution[1] * 0.5) < 0.25 * self._resolution[1] and \
                    ti.abs(r - self._resolution[2] * 0.5) < 0.25 * self._resolution[2]:
                self.m_rho[i, j, r] = self._rho_liquid
                self.m_density[i, j, r] = self._density_liquid
            else:
                self.m_rho[i, j, r] = self._rho_gas
                self.m_density[i, j, r] = self._density_gas
            temp_density = b * self.m_density[i, j, r] / 4.0
            self.m_pressure[i, j, r] = self.m_density[i, j, r] * RT * temp_density * \
                                       (4.0 - 2.0 * temp_density) / ti.pow((1 - temp_density), 3) - \
                                       a * self.m_density[i, j, r] * self.m_density[i, j, r] + \
                                       self.m_density[i, j, r] * RT
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        for I in ti.grouped(self.m_rho):
            for rr in ti.static(range(self._dim)):
                u_tackle[rr] = self.m_velocity[I, rr]
            self.cal_feq(u_tackle, I)
            for k in ti.static(range(self._Q)):
                self.m_f[I, k] = self.m_feq[I, k]

            self.cal_geq(u_tackle, I)
            for k in ti.static(range(self._Q)):
                self.m_g[I, k] = self.m_feq[I, k]

    @ti.func
    def run_step(self):
        self.streaming()
        self.macro_variable_process()
        self.cal_forces()
        self.rebounce_f()
        self.rebounce_g()
        self.collision()

        self.m_time[None] += self._dt
        self.compute_image()
        # self.error_check()

    # @ti.func
    # def cal_feq(self, vel, I):
    #     u2 = 0.0
    #     for rr in range(self._dim):
    #         u2 += self.m_velocity[I, rr] * self.m_velocity[I, rr]
    #     for k in range(self._Q):
    #         eu = 0.0
    #         for rr in range(self._dim):
    #             eu += self._e[k, rr] * self.m_velocity[I, rr]
    #         self.m_feq[I, k] = self.m_rho[I] * self._w[k] * (
    #                 1.0 + eu / self._cs2 + 0.5 * eu * eu / self._cs2 / self._cs2 - 0.5 * u2 / self._cs2)
            # self.m_feq[I, k] = self.m_rho[I] * self._w[k]
            # for rr in ti.static(range(self._dim)):
            #     self.m_feq[I, k] *= (2.0 - ti.sqrt(1.0 + 3.0 * vel[rr] * vel[rr]))
            #     self.m_feq[I, k] *= ti.pow((2.0 * vel[rr] + ti.sqrt(1.0 + 3.0 * vel[rr] * vel[rr])) / (1.0 - vel[rr]),
            #                                self._e[k, rr])

    @ti.func
    def cal_geq(self, vel, I):
        for k in ti.static(range(self._Q)):
            feq_temp = self.m_feq[I, k] / self._w[k] / self.m_rho[I] - 1.0
            self.m_feq[I, k] = self._w[k] * (self.m_pressure[I] + self._cs2 * self.m_density[I] * feq_temp)

    @ti.func
    def streaming(self):
        wid, hei, dep = self.m_rho.shape
        for i, j, r in self.m_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0

            j_front = j - 1 if j > 0 else hei - 1
            j_behind = j + 1 if j < hei - 1 else 0

            r_down = r - 1 if r > 0 else dep - 1
            r_up = r + 1 if r < dep - 1 else 0

            self.m_feq[i, j, r, 0] = self.m_f[i, j, r, 0]

            self.m_feq[i_right, j, r, 1] = self.m_f[i, j, r, 1]
            self.m_feq[i, j_behind, r, 2] = self.m_f[i, j, r, 2]
            self.m_feq[i_left, j, r, 3] = self.m_f[i, j, r, 3]
            self.m_feq[i, j_front, r, 4] = self.m_f[i, j, r, 4]

            self.m_feq[i_right, j_behind, r, 5] = self.m_f[i, j, r, 5]
            self.m_feq[i_left, j_behind, r, 6] = self.m_f[i, j, r, 6]
            self.m_feq[i_left, j_front, r, 7] = self.m_f[i, j, r, 7]
            self.m_feq[i_right, j_front, r, 8] = self.m_f[i, j, r, 8]

            self.m_feq[i, j, r_up, 9] = self.m_f[i, j, r, 9]

            self.m_feq[i_right, j, r_up, 10] = self.m_f[i, j, r, 10]
            self.m_feq[i, j_behind, r_up, 11] = self.m_f[i, j, r, 11]
            self.m_feq[i_left, j, r_up, 12] = self.m_f[i, j, r, 12]
            self.m_feq[i, j_front, r_up, 13] = self.m_f[i, j, r, 13]

            self.m_feq[i, j, r_down, 14] = self.m_f[i, j, r, 14]

            self.m_feq[i_right, j, r_down, 15] = self.m_f[i, j, r, 15]
            self.m_feq[i, j_behind, r_down, 16] = self.m_f[i, j, r, 16]
            self.m_feq[i_left, j, r_down, 17] = self.m_f[i, j, r, 17]
            self.m_feq[i, j_front, r_down, 18] = self.m_f[i, j, r, 18]

        for I in ti.grouped(self.m_f):
            self.m_f[I] = self.m_feq[I]

        for i, j, r in self.m_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0

            j_front = j - 1 if j > 0 else hei - 1
            j_behind = j + 1 if j < hei - 1 else 0

            r_down = r - 1 if r > 0 else dep - 1
            r_up = r + 1 if r < dep - 1 else 0

            self.m_feq[i, j, r, 0] = self.m_g[i, j, r, 0]

            self.m_feq[i_right, j, r, 1] = self.m_g[i, j, r, 1]
            self.m_feq[i, j_behind, r, 2] = self.m_g[i, j, r, 2]
            self.m_feq[i_left, j, r, 3] = self.m_g[i, j, r, 3]
            self.m_feq[i, j_front, r, 4] = self.m_g[i, j, r, 4]

            self.m_feq[i_right, j_behind, r, 5] = self.m_g[i, j, r, 5]
            self.m_feq[i_left, j_behind, r, 6] = self.m_g[i, j, r, 6]
            self.m_feq[i_left, j_front, r, 7] = self.m_g[i, j, r, 7]
            self.m_feq[i_right, j_front, r, 8] = self.m_g[i, j, r, 8]

            self.m_feq[i, j, r_up, 9] = self.m_g[i, j, r, 9]

            self.m_feq[i_right, j, r_up, 10] = self.m_g[i, j, r, 10]
            self.m_feq[i, j_behind, r_up, 11] = self.m_g[i, j, r, 11]
            self.m_feq[i_left, j, r_up, 12] = self.m_g[i, j, r, 12]
            self.m_feq[i, j_front, r_up, 13] = self.m_g[i, j, r, 13]

            self.m_feq[i, j, r_down, 14] = self.m_g[i, j, r, 14]

            self.m_feq[i_right, j, r_down, 15] = self.m_g[i, j, r, 15]
            self.m_feq[i, j_behind, r_down, 16] = self.m_g[i, j, r, 16]
            self.m_feq[i_left, j, r_down, 17] = self.m_g[i, j, r, 17]
            self.m_feq[i, j_front, r_down, 18] = self.m_g[i, j, r, 18]

        for I in ti.grouped(self.m_f):
            self.m_g[I] = self.m_feq[I]

    @ti.func
    def rebounce_g(self):
        for I in ti.grouped(self.m_cell_type):
            if self.m_cell_type[I] == 1:
                self.m_g[I, 1], self.m_g[I, 3] = self.m_g[I, 3], self.m_g[I, 1]
                self.m_g[I, 2], self.m_g[I, 4] = self.m_g[I, 4], self.m_g[I, 2]
                self.m_g[I, 5], self.m_g[I, 7] = self.m_g[I, 7], self.m_g[I, 5]
                self.m_g[I, 6], self.m_g[I, 8] = self.m_g[I, 8], self.m_g[I, 6]

                self.m_g[I, 9], self.m_g[I, 14] = self.m_g[I, 14], self.m_g[I, 9]

                self.m_g[I, 10], self.m_g[I, 17] = self.m_g[I, 17], self.m_g[I, 10]
                self.m_g[I, 11], self.m_g[I, 18] = self.m_g[I, 18], self.m_g[I, 11]
                self.m_g[I, 12], self.m_g[I, 15] = self.m_g[I, 15], self.m_g[I, 12]
                self.m_g[I, 13], self.m_g[I, 16] = self.m_g[I, 16], self.m_g[I, 13]

    @ti.func
    def macro_variable_process(self):
        b = self._b
        a = self._a
        RT = self._RT
        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                self.m_rho[I] = 0.0
                for k in ti.static(range(self._Q)):
                    self.m_rho[I] += self.m_f[I, k]
            self.m_density[I] = self._density_gas + \
                                (self.m_rho[I] - self._rho_gas) / (self._rho_liquid - self._rho_gas) * \
                                (self._density_liquid - self._density_gas)
            if self.m_cell_type[I] == 0 or self.m_cell_type[I] == 2:
                temp_rho = b * self.m_rho[I] / 4.0
                self.m_prho[I] = self.m_pressure[I] - self._cs2 * self.m_density[I]
                self.m_fai[I] = self.m_rho[I] * RT * \
                                (4 * temp_rho - 2 * temp_rho * temp_rho) / ti.pow((1 - temp_rho), 3) - \
                                a * self.m_rho[I] * self.m_rho[I]
        # calculation of contact angle
        wid, hei, dep = self._resolution
        for i in range(wid):
            for j in range(hei):
                if self._wall[0][4] == 'z' and self._wall[0][5] == 'Z':
                    i_left = i - 1 if i > 0 else wid - 1
                    i_right = i + 1 if i < wid - 1 else 0

                    j_front = j - 1 if j > 0 else hei - 1
                    j_behind = j + 1 if j < hei - 1 else 0

                    # -z surface.
                    temp_angle = ti.sqrt(ti.pow(self.m_rho[i_right, j, 2] - self.m_rho[i_left, j, 2], 2) +
                                         ti.pow(self.m_rho[i, j_behind, 2] - self.m_rho[i, j_front, 2], 2))
                    theta = 0.0
                    if temp_angle < 1e-6 and self.m_rho[i, j, 1] >= self.m_rho[i, j, 3]:
                        theta = 0.0
                    elif temp_angle < 1e-6 and self.m_rho[i, j, 1] < self.m_rho[i, j, 3]:
                        theta = 180.0
                    else:
                        theta = 90 - ti.atan2(temp_angle, (self.m_rho[i, j, 1] - self.m_rho[i, j, 3])) * 180.0 / math.pi

                    if theta > self._theta_A:
                        theta = self._theta_A
                        self.m_rho[i, j, 1] = self.m_rho[i, j, 3] + ti.tan(math.pi * (90.0 - theta) / 180.0) * temp_angle
                    elif theta < self._theta_R:
                        theta = self._theta_R
                        self.m_rho[i, j, 1] = self.m_rho[i, j, 3] + ti.tan(math.pi * (90.0 - theta) / 180.0) * temp_angle

                    # -z surface.
                    temp_angle = ti.sqrt(ti.pow(self.m_rho[i_right, j, hei - 3] - self.m_rho[i_left, j, hei - 3], 2) +
                                         ti.pow(self.m_rho[i, j_behind, hei - 3] - self.m_rho[i, j_front, hei - 3], 2))
                    if temp_angle < 1e-6 and self.m_rho[i, j, hei - 2] >= self.m_rho[i, j, hei - 4]:
                        theta = 0.0
                    elif temp_angle < 1e-6 and self.m_rho[i, j, hei - 2] < self.m_rho[i, j, hei - 4]:
                        theta = 180.0
                    else:
                        theta = 90 - ti.atan2(temp_angle, (self.m_rho[i, j, hei - 2] - self.m_rho[i, j, hei - 4])) * 180.0 / math.pi

                    if theta > self._theta_A:
                        theta = self._theta_A
                        self.m_rho[i, j, hei - 2] = self.m_rho[i, j, hei - 4] + ti.tan(
                            math.pi * (90.0 - theta) / 180.0) * temp_angle
                    elif theta < self._theta_R:
                        theta = self._theta_R
                        self.m_rho[i, j, hei - 2] = self.m_rho[i, j, hei - 4] + ti.tan(
                            math.pi * (90.0 - theta) / 180.0) * temp_angle

                    # ghost area (in m_rho):
                    self.m_rho[i, j, 0] = self.m_rho[i, j, 1]
                    self.m_rho[i, j, hei - 1] = self.m_rho[i, j, hei - 2]

                    # wall area (in fai & prho)
                    self.m_fai[i, j, 1] = self.m_fai[i, j, 2]
                    self.m_fai[i, j, hei - 2] = self.m_fai[i, j, hei - 3]
                    self.m_prho[i, j, 1] = self.m_prho[i, j, 2]
                    self.m_prho[i, j, hei - 2] = self.m_prho[i, j, hei - 3]

        for i, j, r in self.m_laplacian_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0

            j_front = j - 1 if j > 0 else hei - 1
            j_behind = j + 1 if j < hei - 1 else 0

            r_down = r - 1 if r > 0 else dep - 1
            r_up = r + 1 if r < dep - 1 else 0
            if self._wall[0][0] == "x" and i == 0:
                i_left = i
            if self._wall[0][1] == "X" and i == wid - 1:
                i_right = i
            if self._wall[0][2] == "y" and j == 0:
                j_front = j
            if self._wall[0][3] == "Y" and j == hei - 1:
                j_behind = j
            if self._wall[0][4] == "z" and r == 0:
                r_down = r
            if self._wall[0][5] == "Z" and r == dep - 1:
                r_up = r
            self.m_laplacian_rho[i, j, r] = (self.m_rho[i_left, j, r] + self.m_rho[i_right, j, r] +
                                             self.m_rho[i, j_front, r] + self.m_rho[i, j_behind, r] +
                                             self.m_rho[i, j, r_up] + self.m_rho[i, j, r_down]) * 2.0 / 6.0
            self.m_laplacian_rho[i, j, r] += (self.m_rho[i_left, j_front, r] + self.m_rho[i_right, j_front, r] +
                                              self.m_rho[i_left, j_behind, r] + self.m_rho[i_right, j_behind, r]) / 6.0
            self.m_laplacian_rho[i, j, r] += (self.m_rho[i_left, j, r_down] + self.m_rho[i_right, j, r_down] +
                                              self.m_rho[i_left, j, r_up] + self.m_rho[i_right, j, r_up]) / 6.0
            self.m_laplacian_rho[i, j, r] += (self.m_rho[i, j_front, r_down] + self.m_rho[i, j_behind, r_down] +
                                              self.m_rho[i, j_front, r_up] + self.m_rho[i, j_behind, r_up]) / 6.0
            self.m_laplacian_rho[i, j, r] -= 24.0 * self.m_rho[i, j, r] / 6.0

    @ti.func
    def cal_forces(self):
        wid, hei, dep = self._resolution
        for i, j, r in self.m_laplacian_rho:
            i_left = i - 1 if i > 0 else wid - 1
            i_right = i + 1 if i < wid - 1 else 0

            j_front = j - 1 if j > 0 else hei - 1
            j_behind = j + 1 if j < hei - 1 else 0

            r_down = r - 1 if r > 0 else dep - 1
            r_up = r + 1 if r < dep - 1 else 0
            if self.m_cell_type[i, j, r] != 1:
                self.m_force[i, j, r, 0] = 2.0 * (self.m_laplacian_rho[i_right, j, r] -
                                                  self.m_laplacian_rho[i_left, j, r])
                self.m_force[i, j, r, 0] += (self.m_laplacian_rho[i_right, j_behind, r] -
                                             self.m_laplacian_rho[i_left, j_front, r])
                self.m_force[i, j, r, 0] += (self.m_laplacian_rho[i_right, j_front, r] -
                                             self.m_laplacian_rho[i_left, j_behind, r])
                self.m_force[i, j, r, 0] += (self.m_laplacian_rho[i_right, j, r_up] -
                                             self.m_laplacian_rho[i_left, j, r_down])
                self.m_force[i, j, r, 0] += (self.m_laplacian_rho[i_right, j, r_down] -
                                             self.m_laplacian_rho[i_left, j, r_up])
                self.m_force[i, j, r, 0] *= self._kappa * self.m_rho[i, j, r] / 12.0

                self.m_force[i, j, r, 1] = 2.0 * (self.m_laplacian_rho[i, j_behind, r] -
                                                  self.m_laplacian_rho[i, j_front, r])
                self.m_force[i, j, r, 1] += (self.m_laplacian_rho[i_right, j_behind, r] -
                                             self.m_laplacian_rho[i_left, j_front, r])
                self.m_force[i, j, r, 1] += (self.m_laplacian_rho[i_left, j_behind, r] -
                                             self.m_laplacian_rho[i_right, j_front, r])
                self.m_force[i, j, r, 1] += (self.m_laplacian_rho[i, j_behind, r_up] -
                                             self.m_laplacian_rho[i, j_front, r_down])
                self.m_force[i, j, r, 1] += (self.m_laplacian_rho[i, j_behind, r_down] -
                                             self.m_laplacian_rho[i, j_front, r_up])
                self.m_force[i, j, r, 1] *= self._kappa * self.m_rho[i, j, r] / 12.0

                self.m_force[i, j, r, 2] = 2.0 * (self.m_laplacian_rho[i, j, r_up] -
                                                  self.m_laplacian_rho[i, j, r_down])
                self.m_force[i, j, r, 2] += (self.m_laplacian_rho[i_right, j, r_up] -
                                             self.m_laplacian_rho[i_left, j, r_down])
                self.m_force[i, j, r, 2] += (self.m_laplacian_rho[i_left, j, r_up] -
                                             self.m_laplacian_rho[i_right, j, r_down])
                self.m_force[i, j, r, 2] += (self.m_laplacian_rho[i, j_behind, r_up] -
                                             self.m_laplacian_rho[i, j_front, r_down])
                self.m_force[i, j, r, 2] += (self.m_laplacian_rho[i, j_front, r_up] -
                                             self.m_laplacian_rho[i, j_behind, r_down])
                self.m_force[i, j, r, 2] *= self._kappa * self.m_rho[i, j, r] / 12.0

                self.m_dfai[i, j, r, 0] = 2.0 * (self.m_fai[i_right, j, r] - self.m_fai[i_left, j, r]) + \
                                          (self.m_fai[i_right, j_behind, r] - self.m_fai[i_left, j_front, r]) + \
                                          (self.m_fai[i_right, j_front, r] - self.m_fai[i_left, j_behind, r]) + \
                                          (self.m_fai[i_right, j, r_up] - self.m_fai[i_left, j, r_down]) + \
                                          (self.m_fai[i_right, j, r_down] - self.m_fai[i_left, j, r_up])
                self.m_dfai[i, j, r, 0] /= 12.0

                self.m_dfai[i, j, r, 1] = 2.0 * (self.m_fai[i, j_behind, r] - self.m_fai[i, j_front, r]) + \
                                          (self.m_fai[i_right, j_behind, r] - self.m_fai[i_left, j_front, r]) + \
                                          (self.m_fai[i_left, j_behind, r] - self.m_fai[i_right, j_front, r]) + \
                                          (self.m_fai[i, j_behind, r_up] - self.m_fai[i, j_front, r_down]) + \
                                          (self.m_fai[i, j_behind, r_down] - self.m_fai[i, j_front, r_up])
                self.m_dfai[i, j, r, 1] /= 12.0

                self.m_dfai[i, j, r, 2] = 2.0 * (self.m_fai[i, j, r_up] - self.m_fai[i, j, r_down]) + \
                                          (self.m_fai[i_right, j, r_up] - self.m_fai[i_left, j, r_down]) + \
                                          (self.m_fai[i_left, j, r_up] - self.m_fai[i_right, j, r_down]) + \
                                          (self.m_fai[i, j_behind, r_up] - self.m_fai[i, j_front, r_down]) + \
                                          (self.m_fai[i, j_front, r_up] - self.m_fai[i, j_behind, r_down])
                self.m_dfai[i, j, r, 2] /= 12.0

                self.m_dprho[i, j, r, 0] = 2.0 * (self.m_prho[i_right, j, r] - self.m_prho[i_left, j, r]) + \
                                           (self.m_prho[i_right, j_behind, r] - self.m_prho[i_left, j_front, r]) + \
                                           (self.m_prho[i_right, j_front, r] - self.m_prho[i_left, j_behind, r]) + \
                                           (self.m_prho[i_right, j, r_up] - self.m_prho[i_left, j, r_down]) + \
                                           (self.m_prho[i_right, j, r_down] - self.m_prho[i_left, j, r_up])
                self.m_dprho[i, j, r, 0] /= 12.0

                self.m_dprho[i, j, r, 1] = 2.0 * (self.m_prho[i, j_behind, r] - self.m_prho[i, j_front, r]) + \
                                           (self.m_prho[i_right, j_behind, r] - self.m_prho[i_left, j_front, r]) + \
                                           (self.m_prho[i_left, j_behind, r] - self.m_prho[i_right, j_front, r]) + \
                                           (self.m_prho[i, j_behind, r_up] - self.m_prho[i, j_front, r_down]) + \
                                           (self.m_prho[i, j_behind, r_down] - self.m_prho[i, j_front, r_up])
                self.m_dprho[i, j, r, 1] /= 12.0

                self.m_dprho[i, j, r, 2] = 2.0 * (self.m_prho[i, j, r_up] - self.m_prho[i, j, r_down]) + \
                                           (self.m_prho[i_right, j, r_up] - self.m_prho[i_left, j, r_down]) + \
                                           (self.m_prho[i_left, j, r_up] - self.m_prho[i_right, j, r_down]) + \
                                           (self.m_prho[i, j_behind, r_up] - self.m_prho[i, j_front, r_down]) + \
                                           (self.m_prho[i, j_front, r_up] - self.m_prho[i, j_behind, r_down])
                self.m_dprho[i, j, r, 2] /= 12.0

        for I in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                for rr in ti.static(range(self._dim)):
                    self.m_velocity[I, rr] = 0.0
                    for k in ti.static(range(self._Q)):
                        self.m_velocity[I, rr] += self._e[k, rr] * self.m_g[I, k]
                    self.m_velocity[I, rr] *= self._c
                    self.m_velocity[I, rr] += 0.5 * self._dt * self._RT * self.m_force[I, rr]
                    self.m_velocity[I, rr] /= self.m_density[I] * self._RT
            if self.m_cell_type[I] == 0 or self.m_cell_type[I] == 2:
                self.m_pressure[I] = 0.0
                for k in ti.static(range(self._Q)):
                    self.m_pressure[I] += self.m_g[I, k]
                for rr in ti.static(range(self._dim)):
                    self.m_pressure[I] += -0.5 * self.m_velocity[I, rr] * self.m_dprho[I, rr] * self._dt

    @ti.func
    def collision(self):
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        tau_f, tau_g = self._tau_f, self._tau_g
        for I, in ti.grouped(self.m_rho):
            if self.m_cell_type[I] == 0:
                u_tackle[0] = self.m_velocity[I, 0]
                u_tackle[1] = self.m_velocity[I, 1]
                self.cal_feq(u_tackle, I)
                for k in ti.static(range(self._Q)):
                    self.m_f[I, k] = (1.0 - 1.0 / tau_f) * self.m_f[I, k] + self.m_feq[I, k] / tau_f
                    temp_f, temp_g = 0.0, 0.0
                    for rr in ti.static(range(self._dim)):
                        temp_f += (self._e[k, rr] * self._c - self.m_velocity[I, rr]) * \
                                  (-self.m_dfai[I, rr]) * self.m_feq[I, k]
                        temp_g += (self._e[k, rr] * self._c - self.m_velocity[I, rr]) * \
                                   self.m_force[I, rr] * self.m_feq[I, k] / self.m_rho[I]
                        temp_g += (self._e[k, rr] * self._c - self.m_velocity[I, rr]) * \
                                  (-self.m_dprho[I, rr]) * ((self.m_feq[I, k] / self.m_rho[I]) - self._w[k])
                    self.m_f[I, k] += self._dt * (tau_f - 0.5) / tau_f * temp_f / (self._RT * self.m_rho[I])
                    self.m_g[I, k] = (1.0 - 1.0 / tau_g) * self.m_g[I, k] + self._dt * (tau_g - 0.5) / tau_g * temp_g
                self.cal_geq(u_tackle, I)
                for k in ti.static(range(self._Q)):
                    self.m_g[I, k] += self.m_feq[I, k] / tau_g
