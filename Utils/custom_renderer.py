import torch
import numpy as np
import json
from pathlib import Path
import glob
import imageio.v3 as iio
import imageio.v2 as imageio
from dataclasses import dataclass
from gsplat.rendering import rasterization
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple, Optional
import cv2
from pycolmap import SceneManager
import os
from typing_extensions import assert_never
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    position: torch.Tensor  # [3,] camera position in world space
    look_at: torch.Tensor  # [3,] point the camera is looking at
    up: torch.Tensor  # [3,] up vector
    fov: float  # vertical field of view in degrees
    aspect: float  # aspect ratio (width/height)
    width: int  # image width
    height: int  # image height

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

class ColmapParser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = True,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


def get_camera_params(parser, camera_idx: int = 0) -> Tuple[CameraParams, str]:


    index = camera_idx
    image = imageio.imread(parser.image_paths[index])[..., :3]
    camera_id = parser.camera_ids[index]
    K = parser.Ks_dict[camera_id].copy()  # undistorted K
    params = parser.params_dict[camera_id]
    camtoworlds = parser.camtoworlds[index]

    if len(params) > 0:
        # Images are distorted. Undistort them.
        mapx, mapy = (
            parser.mapx_dict[camera_id],
            parser.mapy_dict[camera_id],
        )
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = parser.roi_undist_dict[camera_id]
        image = image[y : y + h, x : x + w]

    
    # Get camera matrices
    c2w = camtoworlds
    width, height = parser.imsize_dict[camera_id]
    
    # Extract camera parameters
    position = c2w[:3, 3]
    forward = c2w[:3, 2]  # Get forward direction (third column)
    up = -c2w[:3, 1]      # Get up direction (negative second column)
    
    # # Normalize vectors
    # forward = forward / np.linalg.norm(forward)
    # up = up / np.linalg.norm(up)
    
    # Compute look-at point
    look_at = position + forward
    
    # Calculate FOV from intrinsics
    focal_length = K[1, 1]  # Use y-axis focal length
    fov = float(np.degrees(2 * np.arctan(height / (2 * focal_length))))
    
    # Create camera parameters
    camera_params = CameraParams(
        position=torch.tensor(position, dtype=torch.float32),
        look_at=torch.tensor(look_at, dtype=torch.float32),
        up=torch.tensor(up, dtype=torch.float32),
        fov=fov,
        aspect=width/height,
        width=width,
        height=height
    )
    
    return camera_params, parser.image_names[index], torch.tensor(image)

def load_colmap_camera(data_dir: str, camera_idx: int = 0) -> Tuple[CameraParams, str]:
    """
    Load camera parameters from COLMAP data.
    
    Args:
        data_dir: Path to COLMAP data directory
        camera_idx: Index of camera to load
        
    Returns:
        Tuple of (CameraParams, image_filename)
    """
    parser = ColmapParser(data_dir=data_dir, factor=1, normalize=True)
    return get_camera_params(parser, camera_idx)

def load_gaussian_ckpt(
    ckpt_paths: Union[str, List[str]], 
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Load Gaussian parameters from checkpoint file(s)."""
    # Handle different input types
    if isinstance(ckpt_paths, str):
        path = Path(ckpt_paths)
        if path.is_dir():
            ckpt_files = sorted(glob.glob(str(path / "*.pt")))
        else:
            ckpt_files = [ckpt_paths]
    else:
        ckpt_files = ckpt_paths
        
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found at {ckpt_paths}")
        
    print(f"Loading {len(ckpt_files)} checkpoint file(s)...")
    
    # Load checkpoints
    ckpts = [
        torch.load(f, map_location='cpu', weights_only=True)
        for f in ckpt_files
    ]
    
    # Extract and concatenate parameters
    gaussian_params = {}
    param_keys = ckpts[0]['splats'].keys()
    
    for key in param_keys:
        params = [ckpt['splats'][key] for ckpt in ckpts]
        gaussian_params[key] = torch.cat(params, dim=0).to(device)
        
    print(f"Loaded {len(gaussian_params['means'])} Gaussians")
    return gaussian_params

def get_camera_matrix(params: CameraParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute camera-to-world and intrinsic matrices from parameters."""
    device = params.position.device
    
    # Compute camera axes
    forward = F.normalize(params.look_at - params.position, dim=0)
    up = params.up
    right = F.normalize(torch.linalg.cross(up, forward), dim=0)
    up = torch.linalg.cross(forward, right)
    
    # Build rotation matrix
    rotation = torch.stack([-right, -up, forward], dim=1)
    
    # Build camera-to-world matrix
    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = params.position
    
    # Build camera intrinsics
    focal_length = params.height / (2 * np.tan(np.radians(params.fov) / 2))
    px = params.width / 2
    py = params.height / 2
    
    intrinsics = torch.eye(3, device=device, dtype=torch.float32)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = px
    intrinsics[1, 2] = py
    
    return c2w, intrinsics

def render_novel_view(
    gaussians: Dict[str, torch.Tensor],
    camera_params: CameraParams,
    rasterizer: callable = rasterization,
    sh_degree: int = 3,
    near_plane: float = 0.01,
    far_plane: float = 1000.0,
    radius_clip: float = 3.0,
    packed: bool = False,
    camera_model: str = "pinhole",
    antialiased: bool = False,
) -> torch.Tensor:
    """Render a novel view of Gaussian Splatting representation."""
    device = gaussians["means"].device
    
    # Get camera matrices
    c2w, K = get_camera_matrix(camera_params)
    c2w = c2w.to(device)
    K = K.to(device)
    
    # Process Gaussian parameters
    means = gaussians["means"]
    quats = gaussians["quats"]
    scales = torch.exp(gaussians["scales"])
    opacities = torch.sigmoid(gaussians["opacities"])
    colors = torch.cat([gaussians["sh0"], gaussians.get("shN", torch.zeros(0).to(device))], 1)
    
    # Render
    renders, alphas, _ = rasterizer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(c2w[None]),
        Ks=K[None],
        width=camera_params.width,
        height=camera_params.height,
        sh_degree=sh_degree,
        render_mode='RGB',
        packed=packed,
        radius_clip=radius_clip,
        rasterize_mode="antialiased" if antialiased else "classic",
        near_plane=near_plane,
        far_plane=far_plane,
        camera_model=camera_model,
    )
    
    return torch.clamp(renders[0, ..., :3], 0.0, 1.0)

def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a tensor as an image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = (tensor.cpu().numpy() * 255).astype(np.uint8)
    iio.imwrite(path, img)
    print(f"Saved image to {path}")

def render_gaussian_image(
    ckpt_path: str,
    output_path: str,
    camera_params: Optional[CameraParams] = None,
    colmap_data_dir: Optional[str] = None,
    colmap_camera_idx: int = 0,
    camera_position: Optional[List[float]] = None,
    look_at: Optional[List[float]] = [0, 0, 0],
    up: Optional[List[float]] = [0, 1, 0],
    fov: float = 45.0,
    width: int = 800,
    height: int = 800,
    device: str = "cuda",
    debug: bool = True
) -> None:
    """
    Render a Gaussian Splatting view using either:
    1. Provided CameraParams
    2. COLMAP camera parameters
    3. Manual camera parameters
    
    Args:
        ckpt_path: Path to checkpoint file or directory
        output_path: Where to save the rendered image
        camera_params: Optional pre-configured CameraParams
        colmap_data_dir: Optional path to COLMAP data directory
        colmap_camera_idx: Camera index to use from COLMAP data
        camera_position: Optional manual camera position
        look_at: Optional manual look-at point
        up: Optional manual up vector
        fov: Vertical field of view in degrees (used with manual params)
        width: Image width (used with manual params)
        height: Image height (used with manual params)
        device: Compute device
        debug: Whether to print debug information
    """
    try:
        # Load Gaussians
        if debug:
            print(f"\nLoading checkpoint from: {ckpt_path}")
        gaussian_params = load_gaussian_ckpt(ckpt_path, device=device)
        
        # Get camera parameters based on provided inputs
        if camera_params is not None:
            # Use provided camera parameters
            pass
        elif colmap_data_dir is not None:
            # Load from COLMAP
            camera_params, image_filename, image_gt = load_colmap_camera(
                colmap_data_dir, 
                colmap_camera_idx
            )
            # Update output path with original image name if not specified
            if output_path.endswith('/'):
                output_path = str(Path(output_path) / image_filename)
        elif camera_position is not None:
            # Use manual parameters
            camera_params = CameraParams(
                position=torch.tensor(camera_position, device=device, dtype=torch.float32),
                look_at=torch.tensor(look_at, device=device, dtype=torch.float32),
                up=torch.tensor(up, device=device, dtype=torch.float32),
                fov=fov,
                aspect=width / height,
                width=width,
                height=height
            )
        else:
            raise ValueError(
                "Must provide either camera_params, colmap_data_dir, "
                "or manual camera parameters"
            )
        
        # Move camera params to device
        camera_params.position = camera_params.position.to(device)
        camera_params.look_at = camera_params.look_at.to(device)
        camera_params.up = camera_params.up.to(device)
        
        if debug:
            print(f"\nCamera settings:")
            print(f"Position: {camera_params.position}")
            print(f"Look at: {camera_params.look_at}")
            print(f"Up: {camera_params.up}")
            print(f"FOV: {camera_params.fov}°")
            print(f"Resolution: {camera_params.width}x{camera_params.height}")
            
            # Get camera matrices for debugging
            c2w, K = get_camera_matrix(camera_params)
            print("\nCamera matrices:")
            print("Camera-to-world:\n", c2w.cpu().numpy())
            print("Intrinsics:\n", K.cpu().numpy())
        
        # Render
        if debug:
            print("\nRendering...")
        image = render_novel_view(
            gaussians=gaussian_params,
            camera_params=camera_params,
            sh_degree=3,
            camera_model="pinhole",
            antialiased=True
        )
        
        # metrics:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        ).to(device)
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        image_gt = image_gt.permute(2, 0, 1).unsqueeze(0).to(device)
        image_gt = image_gt.float()/255

        print(f"ssim: {ssim(image, image_gt)}")
        print(f"psnr: {psnr(image, image_gt)}")
        print(f"lpips: {lpips(image, image_gt)}")


        if debug:
            print(f"Rendered image shape: {image.shape}")
            print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Save image
        image = image.squeeze(0).permute(1,2,0)
        save_image(image, output_path)


        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage with COLMAP parameters
    render_gaussian_image(
        ckpt_path="../results/dtu/ckpts/ckpt_6999_rank0.pt",
        output_path="../results/dtu/render_images/",  # Will use original image name
        colmap_data_dir="../data/dtu/scan6/",
        colmap_camera_idx=41,
        debug=False
    )
    
    # # Example usage with manual parameters
    # render_gaussian_image(
    #     ckpt_path="../results/garden/ckpts/ckpt_3499_rank0.pt",
    #     output_path="../results/garden/render_images/manual_view.jpg",
    #     camera_position=[10, 10, 10],
    #     s=[0.1, 0.2, 0],
    #     up=[0, 1, 0],
    #     fov=45.0,
    #     width=800,
    #     height=800,
    #     debug=True
    # )