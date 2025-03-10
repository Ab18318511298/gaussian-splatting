import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui, imgui_ctx
from viewer import Viewer
from viewer.types import ViewerMode
from viewer.widgets.image import NumpyImage
from viewer.widgets.cameras.fps import FPSCamera
from viewer.widgets.viewport_3d import Viewport3D

class GaussianViewer(Viewer):
    def __init__(self, mode: ViewerMode):
        super().__init__(mode)
        self.gaussian_lock = Lock()


    def import_server_modules(self):
        global torch
        import torch

        global GaussianModel
        from scene import GaussianModel

        global PipelineParams, ModelParams
        from arguments import PipelineParams, ModelParams

        global MiniCam
        from scene.cameras import MiniCam

        global render
        from gaussian_renderer import render

    @classmethod
    def from_ply(cls, ply_path, mode: ViewerMode):
        # TODO: Read arguments
        viewer = cls(mode)
        viewer.gaussians = GaussianModel()
        viewer.gaussians.load_ply(ply_path)
        viewer.separate_sh = False
        return viewer
    
    @classmethod
    def from_gaussians(cls, dataset, pipe, gaussians, separate_sh, test_cam, mode: ViewerMode):
        viewer = cls(mode)
        viewer.dataset = dataset
        viewer.pipe = pipe
        viewer.gaussians = gaussians
        viewer.separate_sh = separate_sh
        viewer.test_cam = test_cam
        return viewer

    def create_widgets(self):
        camera = FPSCamera(self.mode, 1280, 720, 30, 0.001, 100)
        img = NumpyImage(self.mode)
        self.point_view = Viewport3D(self.mode, "Point View", img, camera)
        self.scaling_modifier = 1.0

    def step(self):
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        camera = self.point_view.camera
        world_to_view = torch.from_numpy(camera.to_camera).cuda().transpose(0, 1)
        full_proj_transform = torch.from_numpy(camera.full_projection).cuda().transpose(0, 1)
        camera = MiniCam(camera.res_x, camera.res_y, camera.fov_y, camera.fov_x, camera.z_near, camera.z_far, world_to_view, full_proj_transform)
        with self.gaussian_lock:
            net_image = render(camera, self.gaussians, self.pipe, background, scaling_modifier=self.scaling_modifier, use_trained_exp=self.dataset.train_test_exp, separate_sh=self.separate_sh)["render"]
        net_image = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        self.point_view.step(net_image)
    
    def show_gui(self):
        with imgui_ctx.begin("Point View"):
            self.point_view.show_gui()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["local", "client", "server"])
    args = parser.parse_args()

    match args.mode:
        case "local":
            mode = ViewerMode.LOCAL
        case "client":
            mode = ViewerMode.CLIENT
        case "server":
            mode = ViewerMode.SERVER

    if mode is ViewerMode.CLIENT:
        viewer = GaussianViewer(mode)
        viewer.run()
    # viewer = GaussianViewer.from_ply
