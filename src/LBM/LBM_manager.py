import taichi as ti
from .LBM_abstract import LBMAbstract
from .LBM_2d import LBM_2D
from .LBM_2d_KBC import LBM_2D_KBC
from .LBM_2d_SC import LBM_2D_SC
from .LBM_3d_SC import LBM_3D_SC
from .LBM_3d_HCZ import LBM_3D_HCZ



@ti.data_oriented
class LBMManager:
    def __init__(self, core_manager, simulation_parameters, case_parameters):
        self._core_manager = core_manager
        self._log_factory = core_manager._log_factory

        self._case = simulation_parameters["case"]
        self._method = case_parameters[self._case]["method"]

        if simulation_parameters['target'] != 'fluid':
            raise RuntimeError("Only Fluid Simulation supported")
        self._target = simulation_parameters['target']
        if 'LBM' not in self._method:
            raise RuntimeError("Only LBM Simulation supported")
        if '2D' in self._method or '2d' in self._method:
            self._dim = 2
            self._Q = 9
        elif '3D' in self._method or '3d' in self._method:
            self._dim = 3
            self._Q = 19
        else:
            raise RuntimeError("Only 2D or 3D Simulation supported")

        self._resolution = simulation_parameters["resolution"]
        self._dt = simulation_parameters["dt"]
        self._dx = simulation_parameters["dx"]
        self._tau = simulation_parameters["tau"]
        self._c = self._dx / self._dt
        self._cs2 = self._c * self._c / 3.0
        self._Re = simulation_parameters["Re"]
        self._visc = self._cs2 * (self._tau - 0.5)
        self._Lmax = max(self._resolution)
        self._Vmax = self._Re * self._visc / self._Lmax
        self._Ma = self._visc / self._cs2
        self._total_steps = simulation_parameters["total_steps"]

        self._case_parameters = case_parameters

        if self._dim != len(self._resolution):
            raise RuntimeError("method dimension is not consistent with your specified resolution")
        if self._method == "LBM_2d" or self._method == "LBM_2D":
            self._runner = LBM_2D(self)
        elif self._method == "LBM_2d_KBC" or self._method == "LBM_2D_KBC":
            self._runner = LBM_2D_KBC(self)
        elif self._method == "LBM_2d_SC" or self._method == "LBM_2D_SC":
            self._runner = LBM_2D_SC(self)
        elif self._method == "LBM_3d_SC" or self._method == "LBM_3D_SC":
            self._runner = LBM_3D_SC(self)
        elif self._method == "LBM_3d_HCZ" or self._method == "LBM_3D_HCZ":
            self._runner = LBM_3D_HCZ(self)
        else:
            raise RuntimeError("Cannot identify your simulation method: {}".format(self._method))

        self._log_factory.InfoLog("Re = {}, Vmax = {}".format(self._Re, self._Vmax))

    @ti.func
    def initialization(self):
        self._runner.initialization()

    @ti.kernel
    def run_step(self):
        self._runner.run_step()

    def set_density_as_image(self):
        self._core_manager._gui.set_image(self._runner.m_image)

    def get_image_as_numpy(self):
        return self._runner.m_image.to_numpy()

    def kill(self):
        self._runner.kill()
