import os
import taichi as ti
from .LBM.LBM_manager import LBMManager
from .log_factory.log_factory import LogFactory


ti.init(arch=ti.cuda, kernel_profiler=True)


@ti.data_oriented
class CoreManager:
    def __init__(self, simulation_parameters, case_parameters):
        self.root_path = os.path.abspath(os.curdir)
        self.data_path = os.path.join(self.root_path, 'data')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        print("The root path of our project: ", self.root_path)
        self._log_factory = LogFactory(self, False)
        self.make_video = simulation_parameters['make_video']

        self._lbm_manager = LBMManager(self, simulation_parameters, case_parameters)
        self._gui = ti.GUI('gray-scale image with random values',
                           (simulation_parameters["resolution"][0], simulation_parameters["resolution"][1]))
        self._video_manager = ti.VideoManager(output_dir=self.data_path, framerate=24, automatic_build=False)

    @ti.kernel
    def initialization(self):
        self._log_factory.initialization()
        self._lbm_manager.initialization()

    def run(self):
        for epoch in range(self._lbm_manager._total_steps):
            if self._gui.running:
                self._lbm_manager.run_step()
                self._lbm_manager.set_density_as_image()
                self._gui.show()
                if self.make_video and epoch % 20 == 0:
                    self._video_manager.write_frame(self._lbm_manager.get_image_as_numpy())
        if self.make_video:
            self._video_manager.make_video(gif=False, mp4=True)

    def kill(self):
        self._lbm_manager.kill()
        self._log_factory.kill()
