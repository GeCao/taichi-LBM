import taichi as ti
from .LBM_2d import LBM_2D
from typing import List, TypeVar, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from .LBM_manager import LBMManager


@ti.data_oriented
class LBM_2D_SC(LBM_2D):
    def __init__(self, lbm_manager: "LBMManager"):
        super(LBM_2D_SC, self).__init__(lbm_manager=lbm_manager)
        self.m_psi = ti.field(ti.f32, shape=self._resolution)
        self._rho_wall = self._case_parameters[self._lbm_manager._case]['rho_wall']
        self._rho_liquid = self._case_parameters[self._lbm_manager._case]['rho_liquid']
        self._rho_gas = self._case_parameters[self._lbm_manager._case]['rho_gas']

    @ti.func
    def initialization(self):
        for i, j in self.m_rho:
            if ti.abs(i - self._resolution[0] * 0.5) < 0.25 * self._resolution[0] and \
                    ti.abs(j - self._resolution[1] * 0.5) < 0.25 * self._resolution[1]:
                self.m_rho[i, j] = self._rho_liquid
            else:
                self.m_rho[i, j] = self._rho_gas
        u_tackle = ti.Vector.zero(ti.f32, self._dim)
        for I in ti.grouped(self.m_rho):
            u_tackle[0], u_tackle[1] = self.m_velocity[I, 0], self.m_velocity[I, 1]
            self.cal_feq(u_tackle, I)
        for I in ti.grouped(self.m_f):
            self.m_f[I] = self.m_feq[I]

    @ti.func
    def cal_forces(self):
        R = 1.0
        a = 12 * self._cs2
        b = 4.0
        G1 = -1.0 / 3.0
        Tc = 0.3773 * a / (b * R)
        TT0 = 0.875
        TT = TT0 * Tc

        TT = self._cs2 / R
        for I in ti.grouped(self.m_psi):
            if self.m_cell_type[I] == 0:
                temp_rho = self.m_rho[I] * b / 4.0
                temp_Fxy = R * TT * (1.0 + (4.0 * temp_rho - 2.0 * temp_rho**2) / ti.pow((1.0 - temp_rho), 3)) - \
                           a * temp_rho - self._cs2

                self.m_psi[I] = ti.sqrt(2.0 * self.m_rho[I] * temp_Fxy / G1 / self._cs2)
                self.m_pressure[I] = self.m_rho[I] * self._cs2 + 0.5 * self._cs2 * G1 * self.m_psi[I] * self.m_psi[I]

        temp_rho = self._rho_wall * b / 4.0
        temp_Fxy = R * TT * (1.0 + (4.0 * temp_rho - 2.0 * temp_rho**2) / ti.pow((1.0 - temp_rho), 3)) - \
                   a * temp_rho - self._cs2
        psi_w = ti.sqrt(2.0 * self._rho_wall * temp_Fxy / G1 / self._cs2)

        G1 = -1.0 / 3.0
        for i, j in self.m_psi:
            for k in range(self._dim):
                self.m_force[i, j, k] = 0.0
            if self.m_cell_type[i, j] == 0:
                for k in range(1, self._Q):
                    xp = i + self._e[k, 0]
                    yp = j + self._e[k, 1]
                    if xp < 0:
                        xp = self._resolution[0] - 1
                    elif xp >= self._resolution[0]:
                        xp = 0
                    if yp < 0:
                        yp = self._resolution[1] - 1
                    elif yp >= self._resolution[1]:
                        yp = 0
                    if self.m_cell_type[xp, yp] == 1:
                        self.m_force[i, j, 0] += self._w[k] * self._e[k, 0] * psi_w
                        self.m_force[i, j, 1] += self._w[k] * self._e[k, 1] * psi_w
                    else:
                        self.m_force[i, j, 0] += self._w[k] * self._e[k, 0] * self.m_psi[xp, yp]
                        self.m_force[i, j, 1] += self._w[k] * self._e[k, 1] * self.m_psi[xp, yp]
                self.m_force[i, j, 0] *= (-G1 * self.m_psi[i, j] * self._c)
                self.m_force[i, j, 1] *= (-G1 * self.m_psi[i, j] * self._c)