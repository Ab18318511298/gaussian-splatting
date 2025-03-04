import numpy as np
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui, imgui_ctx
from viewer import Viewer
from viewer.types import ViewerMode
from viewer.widgets import NumpyImage

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
        self.img = NumpyImage(self.mode)
        self.scaling_modifier = 1.0

    def step(self):
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        with self.gaussian_lock:
            net_image = render(self.test_cam, self.gaussians, self.pipe, background, scaling_modifier=self.scaling_modifier, use_trained_exp=self.dataset.train_test_exp, separate_sh=self.separate_sh)["render"]
        net_image = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        self.img.step(net_image)
    
    def show_gui(self):
        with imgui_ctx.begin("Image"):
            self.img.show_gui()
    
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