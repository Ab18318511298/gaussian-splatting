import os
from OpenGL.GL import *
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui_ctx, imgui
from viewer import Viewer
from viewer.types import ViewerMode
from viewer.widgets.image import TorchImage
from viewer.widgets.cameras.fps import FPSCamera
from viewer.widgets.monitor import PerformanceMonitor
from viewer.widgets.ellipsoid_viewer import EllipsoidViewer

class Dummy(object):
    pass

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
    def from_ply(cls, model_path, iter, mode: ViewerMode):
        viewer = cls(mode)

        # Read configuration
        viewer.separate_sh = False
        with open(os.path.join(model_path, "cfg_args")) as f:
            params = f.read()
        params = params[10:-1]
        params = params.split(",")
        params = map(lambda x: x.strip(), params)
        params = { key: value for key, value in map(lambda x: x.split("="), params) }

        dataset = Dummy()
        dataset.white_background = params["white_background"] == "True"
        dataset.sh_degree = int(params["sh_degree"])
        dataset.train_test_exp = params["train_test_exp"] == "True"

        pipe = Dummy()
        pipe.debug = "debug" in params
        pipe.antialiasing = "antialiasing" in params
        pipe.compute_cov3D_python = "compute_cov3D_python" in params
        pipe.convert_SHs_python = "convert_SHs_python" in params

        viewer.gaussians = GaussianModel(dataset.sh_degree)
        ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iter}", "point_cloud.ply")
        viewer.gaussians.load_ply(ply_path)
        viewer.dataset = dataset
        viewer.pipe = pipe

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewer.background = background
        return viewer
    
    @classmethod
    def from_gaussians(cls, dataset, pipe, gaussians, separate_sh, mode: ViewerMode):
        viewer = cls(mode)
        viewer.dataset = dataset
        viewer.pipe = pipe
        viewer.gaussians = gaussians
        viewer.separate_sh = separate_sh
        viewer.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        return viewer

    def create_widgets(self):
        self.camera = FPSCamera(self.mode, 1297, 840, 47, 0.001, 100)
        self.point_view = TorchImage(self.mode)
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)
        self.monitor = PerformanceMonitor(self.mode, ["Render"], add_other=False)

        # Render modes
        self.render_modes = ["Splats", "Ellipsoids"]
        self.render_mode = 0

        # Render settings
        self.scaling_modifier = 1.0

    def step(self):
        camera = self.camera
        world_to_view = torch.from_numpy(camera.to_camera).cuda().transpose(0, 1)
        full_proj_transform = torch.from_numpy(camera.full_projection).cuda().transpose(0, 1)
        camera = MiniCam(camera.res_x, camera.res_y, camera.fov_y, camera.fov_x, camera.z_near, camera.z_far, world_to_view, full_proj_transform)

        if self.ellipsoid_viewer.num_gaussians is None:
            self.ellipsoid_viewer.upload(
                self.gaussians.get_xyz.detach().cpu().numpy(),
                self.gaussians.get_rotation.detach().cpu().numpy(),
                self.gaussians.get_scaling.detach().cpu().numpy(),
                self.gaussians.get_opacity.detach().cpu().numpy(),
                self.gaussians.get_features_dc.detach().cpu().numpy()
            )
        
        if self.render_mode == 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with self.gaussian_lock:
                    net_image = render(camera, self.gaussians, self.pipe, self.background, scaling_modifier=self.scaling_modifier, use_trained_exp=self.dataset.train_test_exp, separate_sh=self.separate_sh)["render"]
                net_image = net_image.permute(1, 2, 0)
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)
        if self.render_mode == 1:
            self.ellipsoid_viewer.step(self.camera)
            render_time = glGetQueryObjectuiv(self.ellipsoid_viewer.query, GL_QUERY_RESULT) / 1e6

        self.monitor.step([render_time])
    
    def show_gui(self):
        with imgui_ctx.begin(f"Point View Settings"):
            _, self.render_mode = imgui.list_box("Render Mode", self.render_mode, self.render_modes)

            imgui.separator_text("Render Settings")
            if self.render_mode == 0:
                _, self.scaling_modifier = imgui.drag_float("Scaling Factor", self.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
            if self.render_mode == 1:
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float("Scaling Factor", self.ellipsoid_viewer.scaling_modifier, v_min=0, v_max=10, v_speed=0.01)
                
                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox("Render Floaters", self.ellipsoid_viewer.render_floaters)
                _, self.ellipsoid_viewer.limit = imgui.drag_float("Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01)

            imgui.separator_text("Camera Settings")
            self.camera.show_gui()

        with imgui_ctx.begin("Point View"):
            if self.render_mode == 0:
                self.point_view.show_gui()
            else:
                self.ellipsoid_viewer.show_gui()

            if imgui.is_item_hovered():
                self.camera.process_mouse_input()
            
            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.camera.process_keyboard_input()
        
        with imgui_ctx.begin("Performance"):
            self.monitor.show_gui()
    
    def client_send(self):
        return None, {
            "scaling_modifier": self.scaling_modifier,
            "render_mode": self.render_mode
        }
    
    def server_recv(self, _, text):
        self.scaling_modifier = text["scaling_modifier"]
        self.render_mode = text["render_mode"]
    
if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("model_path")
    local.add_argument("iter", type=int, default=7000)
    client = subparsers.add_parser("client")
    client.add_argument("--ip", default="localhost")
    client.add_argument("--port", type=int, default=6009)
    server = subparsers.add_parser("server")
    server.add_argument("model_path")
    server.add_argument("iter", type=int, default=7000)
    server.add_argument("--ip", default="localhost")
    server.add_argument("--port", type=int, default=6009)
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
    else:
        viewer = GaussianViewer.from_ply(args.model_path, args.iter, mode)
        viewer.run()
