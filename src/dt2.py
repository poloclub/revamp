import os, PIL, csv
import PIL.ImageOps, PIL.ImageDraw
import numpy as np
import torch as ch
import gc
import time
import contextlib
import mitsuba as mi
import drjit as dr
import warnings
from omegaconf import DictConfig, ListConfig
import logging
from typing import Union
from tqdm import tqdm
from .detectors.factory import load_detector
from .utils.gpu_memory_profile import gpu_mem, NvtxRange

 # --------------------- Loss-resolution downscaling helper ---------------------
def _downscale_for_loss(x: ch.Tensor, max_side: int):
    """Downscale NCHW tensor so that max(H,W) <= max_side. Returns (x_ds, scale)."""
    n, c, h, w = x.shape
    maxdim = max(h, w)
    if maxdim <= max_side:
        return x, 1.0
    scale = max_side / float(maxdim)
    x_ds = ch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    return x_ds, scale
# -----------------------------------------------------------------------------

# Optional: help PyTorch use larger segments and reduce fragmentation
# Build allocator config compatible with the installed torch version
try:
    _ver_parts = ch.__version__.split("+")[0].split(".")
    _major = int(_ver_parts[0]) if len(_ver_parts) > 0 else 0
    _minor = int(_ver_parts[1]) if len(_ver_parts) > 1 else 0
except Exception:
    _major, _minor = 0, 0

_alloc_opts = ["max_split_size_mb:256"]
# 'expandable_segments' is only supported on newer PyTorch versions; gate it.
if (_major, _minor) >= (2, 0):
    _alloc_opts.insert(0, "expandable_segments:True")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ",".join(_alloc_opts))
# ==========================================================================


# created symlink in /nvmescratch/mhull32/robust-models-transfer dir for datasets: ln -s /nvmescratch/mhull32/datasets datasets

# coco_train_metadata = MetadataCatalog.get("coco_2017_train")
# coco_train_dataset_dicts = DatasetCatalog.get("coco_2017_train")
# zebra_image_path = "datasets/coco/train2017/000000000034.jpg"


# def model_input_for_path(image_path)->dict:
#     """Returns DT2 formatted model inputs for a COCO image path"""
#     img_names = [c['file_name'] for c in coco_train_dataset_dicts]
#     idx = img_names.index(image_path)
#     dsm = DatasetMapper(dt2_config, is_train=True, augmentations=[])
#     input = dsm.__call__(coco_train_dataset_dicts[idx])
#     print(input['file_name'])
#     return input


def use_provided_cam_position(scene_file:str,sensor_key:str) -> np.array:
    #     from mitsuba import ScalarTransform4f as T  
    # scene = mi.load_file("scenes/water_scene/water_scene.xml")
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)
    sensor_key_tansform_key = f'{sensor_key}.to_world'
    sensor = np.array([p[sensor_key_tansform_key]])
    return sensor

def gen_cam_positions(z,r,size,randomize_radius=False) -> np.ndarray:
    """
    Generates # cam positions of length (size) in a circle of radius (r) 
    at the given latitude (z) on a sphere.  Think of the z value as the height above/below the object
    you want to render.  

    The sphere is centered at the origin (0,0,0) in the scene.  
    """
    if randomize_radius:
        rand_radius = np.random.uniform(z,2.0)
        r = r*rand_radius
    
    if z > r:
        raise Exception("z value must be less than or equal to the radius of the sphere")
    lat_r = np.sqrt(r**2 - z**2)  # find latitude circle radius
    num_points = np.arange(1,size+1)
    angles = np.array([(2 * np.pi * p / size) for p in num_points])
    vertices = np.array([np.array([np.cos(a)*lat_r, z, np.sin(a)*lat_r]) for a in angles])
    return vertices

def load_sensor_at_position(x, y, z, resx=None, resy=None, spp=None):
    from mitsuba import ScalarTransform4f as T, ScalarPoint3f
    origin   = ScalarPoint3f([x, y, z])
    target   = ScalarPoint3f([0, 0, 0])
    up       = ScalarPoint3f([0, 1, 0])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T().look_at(origin=origin, target=target, up=up),
        'sampler': {
            'type': 'independent',
            'sample_count': spp
        },
        'film': {
            'type': 'hdrfilm',
            'width': resx,
            'height': resy,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })

def generate_cam_positions_for_lats(lats=[], r=None, size=None, resx=None, resy=None, spp=None, reps_per_position=1)->tuple:
    """
    Wrapper function to allow generation of camera angles for any list of arbitrary latitudes
    Note that the latitudes must be some z value within the pos/neg value of the radius in the sphere:
    so: {z | -r <= z <= r}
    
    Returns: tuple of (positions, world_transformed_positions)
    """

    positions = gen_cam_positions(lats[0], r, size)
    for i in range(1,len(lats)):
        p = gen_cam_positions(lats[i], r, size)
        positions = np.concatenate((positions, p), axis=0)
    
    world_transformed_positions = np.array([load_sensor_at_position(p[0], p[1], p[2], resx=resx, resy=resy, spp=spp).world_transform() for p in positions])
    world_transformed_positions = np.repeat(world_transformed_positions, reps_per_position)
    return positions, world_transformed_positions    

def generate_batch_sensor(camera_positions=None, resy=None, resx=None, spp=None, randomize_sensors=False):
    from mitsuba import ScalarTransform4f as T
    # all_pos = gen_cam_positions(lats[0], r, size)
    # for i in range(1,len(lats)):
    #     p = gen_cam_positions(lats[i], r, size)
    #     all_pos = np.concatenate((all_pos, p), axis=0)

    batch_sensor = {
        'type': 'batch',
        'film': {
            'type': 'hdrfilm',
            'width': resx * len(camera_positions),
            'height': resy,
            'sample_border': True,
            'filter': { 'type': 'box' }
        },
        'sampler': {
            'type': 'independent',
            'sample_count': spp
        }
    }

    for i, p in enumerate(camera_positions):
        target_x = 0
        target_y = 0
        target_z = 0
        up_vec_y = 1
        up_vec_z = 0

        if randomize_sensors:
            target_x = np.random.uniform(-0.2, 0.2)
            target_y = np.random.uniform(-0.2, 0.2)
            target_z = np.random.uniform(-0.2, 0.2)
            up_vec_z = np.random.uniform(-1, 1)
            up_vec_y = 1 - np.abs(up_vec_z)

        origin = mi.ScalarPoint3f([p[0], p[1], p[2]])
        batch_sensor[f'sensor{i}'] = {
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T().look_at(
                origin=origin,
                target=mi.ScalarPoint3f([target_x, target_y, target_z]),
                up=mi.ScalarPoint3f([0, up_vec_y, up_vec_z])
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': spp
            },
            'film': {
                'type': 'hdrfilm',
                'width': resx,
                'height': resy,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        }

    return batch_sensor

@dr.wrap(source='drjit', target='torch')
def reshape_batch_imgs(img, n, h, w, c):
    imgs = img.reshape(h, n, w, c).permute(1, 0, 2, 3)
    return imgs

def generate_bboxes_for_target(target_mesh:ListConfig, camera_positions, resx:int, resy:int)->np.ndarray:

    if len(target_mesh)>1:
        warnings.warn("Only 1 target mesh will be rendered.  Cannot generate bboxes for multiple targets.")
        
    # generate a b&w scene with the target mesh w/ black material and white envmap for a 'solo' render.
    scene = mi.load_dict({
        "type": "scene",
        "myintegrator": {
            "type": "path",
        },
        'emitter': {
            'type': 'envmap',
            'filename': "scenes/bw/textures/white.exr",
        },
        "mat-Black":  {
            'type': 'principled',
                'base_color': {
                    'type': 'rgb',
                    'value': [0.0,0.0,0.0]
                },
                'metallic': 0.0,
                'specular': 0.0,
                'roughness': 1.0,
                'spec_tint': 0.0,
                'anisotropic': 0.0,
                'sheen': 0.0,
                'sheen_tint': 0.0,
                'clearcoat': 0.0,
                'clearcoat_gloss': 0.0,
                'spec_trans': 0.0
                ,'id': 'mat-Black'
        },
        'shape': {
            'type': 'ply',
            'filename': target_mesh[0],
            'bsdf': {
                'type': 'ref', 
                'id': 'mat-Black'
            }
        }
    })
    gt_bboxes = []
    
    batch_sensor_dict = generate_batch_sensor(camera_positions=camera_positions, resx=resx, resy=resy, spp=2)
    batch_sensor = mi.load_dict(batch_sensor_dict)
    n, h, w, c = len(camera_positions), resy, resx, 3
    img = mi.render(scene, sensor=batch_sensor)
    imgs = reshape_batch_imgs(img, n, h, w, c)
    for i in range(n):
        # convert the renders to binary images and get bboxes
        bmp = mi.Bitmap(imgs[i]).convert(pixel_format=mi.Bitmap.PixelFormat.RGB,component_format=mi.Struct.Type.UInt8)
        pil_img = PIL.Image.fromarray(np.array(bmp), mode='RGB')
        pil_img = pil_img.convert('L')
        threshold = 128
        # binarizing is important, some artifacts result in pixels that aren't completely black and affects bbox
        pil_img = pil_img.point(lambda p: p > threshold and 255)
        pil_img = PIL.ImageOps.invert(pil_img)
        bbox = pil_img.getbbox()
        gt_bboxes.append(bbox)
        
        # Annotate bbox on the image
        draw = PIL.ImageDraw.Draw(pil_img)
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), "object", fill="red")
        pil_img.save(f'renders/bw/bbox_render_{i}.jpg')        
        
    return np.array(gt_bboxes)

def attack_dt2(cfg:DictConfig) -> None:

    # cuda_visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", default=1)
    # DEVICE = f"cuda:{cuda_visible_device}"
    DEVICE = f"cuda:{cfg.device}" 
    # ch.cuda.set_device(DEVICE)

    # Hydra-controlled memory profiling flag (default false via config)
    MEM_PROFILE = getattr(cfg.sysconfig, "mem_profile", False)

    # If disabled, silence gpu_mem() and NvtxRange by rebinding the GLOBAL symbols
    if not MEM_PROFILE:
        global gpu_mem, NvtxRange
        def _gpu_mem_noop(*args, **kwargs):
            return None
        class _NvtxRangeNoop:
            def __init__(self, *_, **__):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        gpu_mem = _gpu_mem_noop
        NvtxRange = _NvtxRangeNoop
    if ch.cuda.is_available():
        ch.cuda.reset_peak_memory_stats()
        gpu_mem("startup", DEVICE)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logger = logging.getLogger("dt2")

    batch_size = cfg.attack.batch_size
    eps = cfg.attack.eps
    eps_step =  cfg.attack.eps_step
    targeted =  cfg.attack.targeted
    target_class = cfg.attack.target_idx
    target_string = cfg.attack_class
    untargeted_class = cfg.attack.untarget_idx
    untargeted_string = cfg.untargeted_class
    iters = cfg.attack.iters
    spp = cfg.attack.samples_per_pixel
    spp_grad = getattr(cfg.attack, "spp_grad", 1)
    # NOTE:
    #  - `spp_grad` controls the SPP used for the gradient/backprop render. Keep this low (1–2)
    #    to minimize AD memory. We do NOT save these low-SPP images to disk.
    #  - `spp_viz` controls a separate high-quality, no-grad visualization render that *is*
    #    saved as `*_hq.png` in both renders/ and preds/.
    spp_viz = getattr(cfg.attack, "spp_viz", 32)
    multi_pass_rendering = cfg.attack.multi_pass_rendering
    multi_pass_spp_divisor = cfg.attack.multi_pass_spp_divisor
    scene_file = cfg.scene.path
    param_keys = cfg.scene.target_param_keys
    target_mesh = cfg.scene.target_mesh
    sensor_key = cfg.scene.sensor_key
    randomize_sensors = cfg.scenario.randomize_positions 
    scene_file_dir = os.path.dirname(scene_file)
    tex_paths = cfg.scene.textures
    multicam = cfg.multicam
    resx = cfg.scene.resx
    resy = cfg.scene.resy
    use_batch_sensor = cfg.scene.use_batch_sensor
    tmp_perturbation_path = os.path.join(f"{scene_file_dir}",f"textures/{target_string}_tex","tmp_perturbations")
    if os.path.exists(tmp_perturbation_path) == False:
        os.makedirs(tmp_perturbation_path)
    render_path = os.path.join(f"renders",f"{target_string}")
    if os.path.exists(render_path) == False:
        os.makedirs(render_path)
    preds_path = os.path.join("preds",f"{target_string}")
    if os.path.exists(preds_path) == False:
        os.makedirs(preds_path)    
    
    if multi_pass_rendering:
        logger.info(f"Using multi-pass rendering with {spp//multi_pass_spp_divisor} passes")

    # Max side (in pixels) for D2 loss path to control activation memory
    loss_max_side = getattr(cfg.model, "loss_max_side", 512)
    
    # img_name = 'render_39_s3'
    # adv_path = f'/nvmescratch/mhull32/robust-models-transfer/renders/{img_name}.jpg'
    mi.set_variant('cuda_ad_rgb')
    for tex in tex_paths:
        mitsuba_tex = mi.load_dict({
            'type': 'bitmap',
            'id': 'heightmap_texture',
            'filename': tex,
            'raw': True
        })
        mt = mi.traverse(mitsuba_tex)
    # FIXME - allow variant to be set in the configuration.
   
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)    

    k = param_keys

    keep_keys = [k for k in param_keys]
    k1 = f'{sensor_key}.to_world'
    k2 = f'{sensor_key}.film.size'
    keep_keys.append(k1)
    p.keep(keep_keys)
    p.update()
    orig_texs = []

    if multicam == 1:
        moves_matrices = use_provided_cam_position(scene_file=scene_file, sensor_key=sensor_key)  
    else:
        cam_position_matrices, world_transformed_cam_position_matrices = generate_cam_positions_for_lats(lats=cfg.scene.sensor_z_lats \
                                                            ,r=cfg.scene.sensor_radius \
                                                            ,size=cfg.scene.sensor_count \
                                                            ,resx=resx \
                                                            ,resy=resy \
                                                            ,spp=spp)
        if use_batch_sensor: 
            moves_matrices =  cam_position_matrices
            batch_sensor_dict = generate_batch_sensor(moves_matrices, resx, resy, spp)
            batch_sensor = mi.load_dict(batch_sensor_dict)
            sensor_count =  batch_sensor.m_film.size()[0]//resx #divide by resx to get number of sensors
        else: 
            moves_matrices =  world_transformed_cam_position_matrices       
            if randomize_sensors:
                np.random.shuffle(moves_matrices)
            
        gt_bboxes = generate_bboxes_for_target(target_mesh, cam_position_matrices, resx, resy)
    
    #FIXME - truncate some of the camera positions, when we don't want to render an entire orbit
    # moves_matrices = moves_matrices[10:]
    # reverse moves_matrices
    # moves_matrices = moves_matrices[::-1] 
    # concat moves_matrices with moves_matrices[24:]
    # moves_matrices = np.concatenate((moves_matrices[0:5], moves_matrices[25:][::-1]), axis=0)

    # Initialize the selected detector via factory abstraction
    detector = load_detector(cfg)
    detector.load_model()
    # Detector implementations manage their own params; ensure no detector weights require grads if applicable
    try:
        for p in detector.model.parameters():
            p.requires_grad_(False)
    except Exception:
        pass

    def optim_batch(scene, batch_size, camera_positions, spp, k, label, unlabel, iters, alpha, epsilon, targeted=False):
        # run attack
        if targeted:
            assert(label is not None)

        if ch.cuda.is_available():
            ch.cuda.reset_peak_memory_stats()
        gpu_mem("optim_batch:start", DEVICE)

        #assert(batch_size <= sensor_count)
        assert(batch_size <= camera_positions.size)
        success = False
        
        @dr.wrap(source='drjit', target='torch')
        def model_input(x, target, bboxes, batch_size=1):
            """Convert Mitsuba render tensor to detector-ready NCHW and compute loss.
            x comes in as HxWxC (single sensor) or Hx(W*N)xC (batch sensor with N views concatenated width-wise).
            """
            # Ensure x is float in [0,1]
            if not ch.is_floating_point(x):
                x = x.float()

            # Case 1: batch sensor => H x (W*N) x C -> N x C x H x W
            if batch_size > 1:
                assert x.dim() == 3, "Expected Hx(W*N)xC tensor from Mitsuba for batch sensor"
                H, WN, C = x.shape
                assert C == 3, f"Expected 3 color channels, got {C}"
                assert WN % batch_size == 0, f"Width {WN} not divisible by batch size {batch_size}"
                W = WN // batch_size
                # reshape to H x N x W x C, then to N x C x H x W
                x = x.view(H, batch_size, W, C).permute(1, 3, 0, 2).contiguous().requires_grad_()
            else:
                # Case 2: single sensor => H x W x C -> 1 x C x H x W
                if x.dim() == 3 and x.shape[-1] == 3:
                    x = x.permute(2, 0, 1).unsqueeze(0).contiguous().requires_grad_()
                elif x.dim() == 4 and x.shape[1] == 3:
                    # already NCHW with batch dim 1
                    pass
                else:
                    raise RuntimeError(f"Unexpected image tensor shape for single-sensor path: {tuple(x.shape)}")
            x.retain_grad()

            # Optional downscaling to control activation memory (detector-agnostic)
            x, scale = _downscale_for_loss(x, loss_max_side)
            height, width = x.shape[2], x.shape[3]

            # Normalize bbox shape and scale to match downscaled image
            if ch.tensor(bboxes).dim() == 1:
                gt_boxes = ch.tensor(bboxes).unsqueeze(0)
            else:
                gt_boxes = ch.tensor(bboxes)
            if scale != 1.0:
                gt_boxes = gt_boxes * scale

            # Ensure target is a torch.LongTensor on the same device
            if not isinstance(target, ch.Tensor):
                try:
                    # drjit arrays expose `.array`; fall back to numpy conversion
                    t_arr = getattr(target, 'array', None)
                    if t_arr is not None:
                        target = ch.as_tensor(t_arr, dtype=ch.long, device=x.device)
                    else:
                        target = ch.as_tensor(target, dtype=ch.long, device=x.device)
                except Exception:
                    target = ch.as_tensor(target, dtype=ch.long, device=x.device)
            else:
                target = target.to(device=x.device, dtype=ch.long)

            # Delegate to detector for loss computation (expects N x C x H x W in ~[0,1])
            loss = detector.infer(x, target, gt_boxes, batch_size=x.shape[0])
            return loss

        params = mi.traverse(scene)
        for k in param_keys:
            if isinstance(params[k], dr.cuda.ad.TensorXf):
                # use Float if dealing with just texture colors (not a texture map)
                orig_tex = dr.cuda.ad.TensorXf(params[k])
            elif isinstance(params[k], dr.cuda.ad.Float):
                orig_tex = dr.cuda.ad.Float(params[k])        
            else:
                raise Exception("Unrecognized Differentiable Parameter Data Type.  Should be one of dr.cuda.ad.Float or dr.cuda.ad.TensorXf")

            # orig_tex.set_label(f"{k}_orig_tex")
            dr.set_label(orig_tex, f"{k}_orig_tex")
            orig_texs.append(orig_tex)
        
        # indicate sensors to use in producing the perturbation
        # e.g., [0,1,2,3] will use sensors 0-3 focus on Taxi/Cement Truck in 'intersection_taxi.xml'
        # sensor 10 is focused on stop sign.
        sensors = [0]
        if iters % len(sensors) != 0:
            logger.warning("Uneven amount of iterations provided for sensors; some sensors will be used more than others during attack")
        # if only one camera in the scene, then this idx will be repeated for each iter
        camera_idx = ch.Tensor(np.array(sensors)).repeat(int(iters/len(sensors))).to(dtype=ch.uint8).numpy().tolist()
        # one matrix per camera position that we want to render from, equivalent to batch size
        # e.g., batch size of 5 = 5 required camera positions
        
        cam_idx = 0
        for it in range(iters):
            # print(f'iter {it}')
            # logger.info(f"iter {it}")
            # keep 2 sets of parameters because we only need to differentiate wrt texture
            num_cam_positions = sensor_count if use_batch_sensor else camera_positions.size
            diff_params = mi.traverse(scene)
            non_diff_params = mi.traverse(scene)
            diff_params.keep([k for k in param_keys])
            non_diff_params.keep([k1,k2])
            non_diff_params[k2] = [resx, resy]
            # Optimizer is not used but necessary to instantiate to get gradients from diff rendering.
            opt = mi.ad.Adam(lr=0.1, params=diff_params)
            for i,k in enumerate(param_keys):
                dr.enable_grad(orig_texs[i])
                dr.enable_grad(opt[k])
                # opt[k].set_label(f"{k}_bitmap")
                dr.set_label(opt[k], f"{k}_bitmap")
            # sample random camera positions (=batch size) for each batch iteration
            if num_cam_positions > 1:
                np.random.seed(it+1)
                sampled_camera_positions_idx = np.random.randint(low=0, high=num_cam_positions-1,size=batch_size)
            else: sampled_camera_positions_idx = [0]
            if batch_size > 1:
                sampled_camera_positions = camera_positions[sampled_camera_positions_idx]
            else:
                sampled_camera_positions = camera_positions
            if success:
                cam_idx += 1
                logger.info(f"Successful pred, using camera_idx {cam_idx}")
            if use_batch_sensor:
                batch_sensor_film_size = mi.traverse(batch_sensor)["film.size"]
                N, H, W, C = sensor_count, batch_sensor_film_size[1], batch_sensor_film_size[0], 3
            else: 
                N, H, W, C = batch_size, non_diff_params[k2][0], non_diff_params[k2][1], 3
                        
 
                
            for b in range(0, batch_size):

                # EOT Strategy
                # set the camera position, render & attack
                if not use_batch_sensor:
                    if cam_idx > len(sampled_camera_positions)-1:
                        logger.info(f"Successful detections on all {len(sampled_camera_positions)} positions.")
                        return
                    if batch_size > 1: # sample from random camera positions
                        cam_idx = b
                    if isinstance(sampled_camera_positions[cam_idx], mi.cuda_ad_rgb.Transform4f):
                        non_diff_params[k1].matrix = sampled_camera_positions[cam_idx].matrix
                    else:
                        non_diff_params[k1].matrix = mi.cuda_ad_rgb.Matrix4f(sampled_camera_positions[cam_idx])
                non_diff_params.update()
                params.update(opt)            
                
                sensor_i = batch_sensor if use_batch_sensor else camera_idx[it]

                gpu_mem("pre-render", DEVICE)
                if multi_pass_rendering:
                    # For gradient computation, keep it cheap regardless of multi-pass settings
                    render_passes = 1
                    mini_pass_spp = max(1, int(spp_grad))
                    seed = np.random.randint(0, 1000)
                    img = mi.render(scene, params=params, spp=mini_pass_spp, sensor=sensor_i, seed=seed)
                else:
                    img = mi.render(scene, params=params, spp=max(1, int(spp_grad)), sensor=sensor_i, seed=it+1)
                gpu_mem("post-render", DEVICE)
                
                # TODO - place this into a utils function
                # split image into images for number of sensors if needed!
                # split_imgs = []
                # for i in range(len(camera_positions)):
                #     start = i * H
                #     end = (i+1) * H
                #     split_img = img[:,start:end,:]
                #     # mi.util.write_bitmap(f"split_img_{i}.png", data=split_img, write_async=False)   
                #     split_imgs.append(dr.ravel(split_img))
                    
                #  img.set_label(f"image_b{it}_s{b}")
                dr.set_label(img, f"image_b{it}_s{b}")
                # === Low-SPP visualization disabled ===
                # We deliberately do not write the low-SPP (gradient) render to disk to avoid noisy images.
                # Backprop still uses `img` internally. To increase gradient quality, adjust `spp_grad` in config.
                # If you need to debug, uncomment the lines below.
                # rendered_img_path = os.path.join(render_path, f"render_b{it}_p{b}_s{cam_idx}.png")
                # mi.util.write_bitmap(rendered_img_path, data=img, write_async=False)
                # rendered_img_input = dt2_input(rendered_img_path)
                # success = save_adv_image_preds(
                #     model,
                #     dt2_config,
                #     input=rendered_img_input,
                #     instance_mask_thresh=dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                #     target=label,
                #     untarget=unlabel,
                #     is_targeted=targeted,
                #     path=os.path.join(preds_path, f'render_b{it}_s{cam_idx}.png')
                # )
                target = dr.cuda.ad.TensorXf([label], shape=(1,))

            #imgs = dr.cuda.ad.TensorXf(dr.cuda.ad.Float(img),shape=(N, H, W//N, C))
            
            if (dr.grad_enabled(img)==False):
                dr.enable_grad(img)
            selected_bboxes = gt_bboxes if use_batch_sensor else gt_bboxes[cam_idx] # pass single bbox or all bboxes
            loss = model_input(img, target, selected_bboxes, N)
            gpu_mem("post-model_input(loss ready)", DEVICE)
            sensor_loss = f"[PASS {cfg.sysconfig.pass_idx}] iter: {it} sensor pos: {cam_idx}/{len(sampled_camera_positions)}, loss: {str(loss.array[0])[0:7]}"
            logger.info(sensor_loss)
            dr.enable_grad(loss)
            with NvtxRange("backward"):
                t0 = time.time()
                gpu_mem("pre-backward", DEVICE)
                try:
                    ch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    dr.flush_malloc_cache()
                except Exception:
                    pass
                try:
                    detector.zero_grad()
                except Exception:
                    pass
                dr.backward(loss)
                ch.cuda.synchronize()
                gpu_mem("post-backward", DEVICE)
                if MEM_PROFILE:
                    print(f"[TIMING] backward took {time.time() - t0:.3f}s")
            # Explicit graph cleanup to avoid growth across iters
            try:
                # Drop references and gradients we no longer need
                del selected_bboxes
            except Exception:
                pass
            try:
                # Clear PyTorch autograd state tied to this loss
                if isinstance(loss, ch.Tensor):
                    loss.detach_()
                del loss
            except Exception:
                pass
            gc.collect()
            try:
                ch.cuda.empty_cache()
            except Exception:
                pass

            #########################################################################
            # L-INFattack
            # grad = dr.grad(opt[k])
            # tex = opt[k]
            # eta = alpha * dr.sign(grad)
            # if targeted:
            #     eta = -eta
            # tex = tex + eta
            # eta = dr.clip(tex - orig_tex, -epsilon, epsilon)
            # tex = orig_tex + eta
            # tex = dr.clip(tex, 0, 1)
            #########################################################################
            for i, k in enumerate(param_keys):
                gpu_mem(f"update:{k}:start", DEVICE)
                HH, WW  = dr.shape(dr.grad(opt[k]))[0], dr.shape(dr.grad(opt[k]))[1]
                grad = ch.Tensor(dr.grad(opt[k]).array).view((HH, WW, C))
                tex = ch.Tensor(opt[k].array).view((HH, WW, C))
                _orig_tex = ch.Tensor(orig_texs[i].array).view((HH, WW, C))
                l = len(grad.shape) -  1
                g_norm = ch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
                scaled_grad = grad / (g_norm  + 1e-10)
                #if targeted:
                scaled_grad = -scaled_grad
                # step
                tex = tex + scaled_grad * alpha
                delta  = tex - _orig_tex
                # project
                delta =  delta.renorm(p=2, dim=0, maxnorm=epsilon)
                tex = _orig_tex + delta

                # convert back to mitsuba dtypes            
                tex = dr.cuda.ad.TensorXf(tex.to(DEVICE))
                # divide by average brightness
                scaled_img = img / dr.mean(dr.detach(img))
                tex = tex / dr.mean(scaled_img)         
                tex = dr.clip(tex, 0, 1)
                params[k] = tex     
                dr.enable_grad(params[k])
                params.update()
                gpu_mem(f"update:{k}:after-params.update", DEVICE)
                perturbed_tex = mi.Bitmap(params[k])
                                
                mi.util.write_bitmap(os.path.join(tmp_perturbation_path,f"{k}_{it}.png"), data=perturbed_tex, write_async=False)
                if it==(iters-1) and isinstance(params[k], dr.cuda.ad.TensorXf):
                    perturbed_tex = mi.Bitmap(params[k])
                    mi.util.write_bitmap("perturbed_tex_map.png", data=perturbed_tex, write_async=False)
                ch.cuda.empty_cache()

            # Optional high-quality visualization render (no AD tape)
            if int(spp_viz) > int(spp_grad):
                gpu_mem("pre-render-viz", DEVICE)
                try:
                    with dr.suspend_grad():
                        img_viz = mi.render(scene, params=params, spp=int(spp_viz), sensor=sensor_i, seed=it+12345)
                    # Save HQ image using canonical filename (no _hq suffix)
                    hq_path = os.path.join(render_path, f"render_b{it}_p{b}_s{cam_idx}.png")
                    mi.util.write_bitmap(hq_path, data=img_viz, write_async=False)
                    # Also run predictions on the HQ image and save overlay with canonical name
                    rendered_img_input_hq = detector.preprocess_input(hq_path)
                    _thr = getattr(cfg.model, "score_thresh_test", 0.5)
                    img_for_pred = rendered_img_input_hq['image'].float()
                    # Range-aware scaling so both detectors behave:
                    # - If detector expects [0,1] but input is 0–255, divide by 255.
                    # - If detector expects 0–255 but input is [0,1], multiply by 255.
                    maxv = float(img_for_pred.max().item())
                    if getattr(detector, 'expects_unit_input', True):
                        if maxv > 1.5:
                            img_for_pred = (img_for_pred / 255.0).clamp(0.0, 1.0)
                    else:
                        if maxv <= 1.5:
                            img_for_pred = (img_for_pred * 255.0).clamp(0.0, 255.0)
                    success = detector.predict_and_save(
                        image=img_for_pred,
                        path=os.path.join(preds_path, f'render_b{it}_s{cam_idx}.png'),
                        target=label,
                        untarget=unlabel,
                        is_targeted=targeted,
                        threshold=_thr,
                        format="RGB",
                        gt_bbox=gt_bboxes[cam_idx] if not use_batch_sensor else None,
                        result_dict=False,
                        tile_w=resx,
                        tile_h=resy,
                        tiles=(sensor_count if use_batch_sensor else 1)
                    )
                except Exception as _viz_e:
                    logger.warning(f"[VIZ] High-quality render failed: {_viz_e}")
                gpu_mem("post-render-viz", DEVICE)
            ch.cuda.empty_cache()
            try:
                dr.flush_malloc_cache()
            except Exception:
                pass
            gpu_mem(f"iter {it} end", DEVICE)
        gpu_mem("optim_batch:end", DEVICE)
        return scene
    
    # iters = iters  
    samples_per_pixel = spp
    epsilon = eps
    alpha = eps_step #(epsilon / (iters/50))
    label = target_class
    unlabel = untargeted_class
    img = optim_batch(scene\
                      , batch_size=batch_size\
                      , camera_positions =  moves_matrices\
                      , spp=samples_per_pixel\
                      , k=k, label = label\
                      , unlabel=unlabel\
                      , iters=iters\
                      , alpha=alpha\
                      , epsilon=epsilon\
                      , targeted=targeted)
    gpu_mem("attack_dt2:end", DEVICE)
