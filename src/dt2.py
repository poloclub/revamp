import os, json, random, PIL, csv, sys
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import cv2
import itertools 
import copy
import torch as ch
from torchvision import datasets, transforms
from pycocotools.cocoeval import COCOeval
from random import seed
import torchvision.transforms as TT
import torch.nn as nn
from torchvision.io import read_image
import mitsuba as mi
import drjit as dr
import json
import time
import copy
import argparse
import hydra
from omegaconf import DictConfig
import logging
# from detectron2.utils.logger import setup_logger
# setup_logger()

# detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import DatasetMapper
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.modeling.backbone import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.proposal_generator import RPN, build_proposal_generator
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
    find_top_rpn_proposals,
)
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.events import EventStorage
from detectron2.data.detection_utils import read_image
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Instances
import detectron2.data.transforms as T
from detectron2.data.detection_utils import *
from fvcore.transforms.transform import NoOpTransform
from detectron2.utils.file_io import PathManager


# COCO Labels
BUS = 5
STOP_SIGN = 11
SPORTS_BALL = 32
TRAFFIC_LIGHT = 9
TV = 62
PERSON = 00
CAR = 2
TRUCK = 7
MOTORCYCLE = 3




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

def dt2_input(image_path:str)->dict:
    """
    Construct a Detectron2-friendly input for an image
    """
    input = {}
    filename = image_path
    adv_image = read_image(image_path, format="RGB")
    adv_image_tensor = ch.as_tensor(np.ascontiguousarray(adv_image.transpose(2, 0, 1)))

    height = adv_image_tensor.shape[1]
    width = adv_image_tensor.shape[2]
    instances = Instances(image_size=(height,width))
    instances.gt_classes = ch.Tensor([2])
    # taxi bbox
    # instances.gt_boxes = Boxes(ch.tensor([[ 50.9523, 186.4931, 437.6184, 376.7764]]))
    # stop sign bbox
    # instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, height, width]]))
    instances.gt_boxes = Boxes(ch.tensor([[ 162.0, 145.0, 364.0, 324.0]])) # for 512x512 img
    input['image'] = adv_image_tensor    
    input['filename'] = filename
    input['height'] = height
    input['width'] = width
    input['instances'] = instances
    return input

def save_adv_image_preds(model, dt2_config, input, instance_mask_thresh=0.7, target:int=None, format="RGB", path:str=None):
    """
    Helper fn to save the predictions on an adversarial image
    attacked_image:ch.Tensor An attacked image
    instance_mask_thresh:float threshold pred boxes on confidence score
    path:str where to save image
    """ 
    model.train = False
    model.training = False
    model.proposal_generator.training = False
    model.roi_heads.training = False    
    with ch.no_grad():
        adv_outputs = model([input])
        perturbed_image = input['image'].data.permute((1,2,0)).detach().cpu().numpy()
        pbi = ch.tensor(perturbed_image, requires_grad=False).detach().cpu().numpy()
        if format=="BGR":
            pbi = pbi[:, :, ::-1]
        v = Visualizer(pbi, MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]),scale=1.0)
        instances = adv_outputs[0]['instances']
        things = np.array(MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]).thing_classes) # holds class labels
        predicted_classes = things[instances.pred_classes.cpu().numpy().tolist()] 
        print(f'Predicted Class: {predicted_classes}')        
        mask = instances.scores > instance_mask_thresh
        instances = instances[mask]
        out = v.draw_instance_predictions(instances.to("cpu"))
        target_pred_exists = target in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True  
    PIL.Image.fromarray(pred).save(path)
    if target_pred_exists:
        return True
    return False

def generate_stop_sign_approach_cam_moves(sample_size=10) -> np.array:
    """
    Generate an np.ndarray of camera transform matrices
    Read in a set of camera positions that comprise an animation of the camera position
    """
    # grab animation values of the camera generated in blender aand 
    # construct transform matrices for each camera position we want to sample
    scene_path = os.path.join("scenes", "intersection_taxi")
    animations_path = os.path.join(scene_path,"animations", "cam_moves.csv")
    data = csv.reader(open(animations_path))
    # invert values for blender/mitsuba compatability
    moves = np.array([-float(d[0]) for d in data][0:]) 
    sample_moves = np.random.choice(moves, sample_size) #randomly sample cameraa positions (default=10)
    moves_matrices = []
    mat = p[k1].matrix
    for m in sample_moves:
        _mat = mi.cuda_ad_rgb.Matrix4f(mat)
        _mat[3][2]= m # modify the camera z-position
        moves_matrices.append(_mat)
    return np.array(moves_matrices)

def generate_taxi_cam_positions() -> np.array:
    def load_sensor(r, y, phi, theta):
        from mitsuba import ScalarTransform4f as T
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 1, 0], phi).rotate([0, 0, 1], theta) @ mi.ScalarPoint3f([0, y, r])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, -0.20, 0],
                #up=[0, 0, 1]
                up=[0, 1, 0]            
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
    
    # e.g, use sensor_count=6 & phis=30 to get 180 deg view
    sensor_count = 6
    radius = 5.0
    phis = [60.0 * i for i in range(sensor_count)]
    theta = 12.0
    # y = 5
    # ys = [1,3,5]
    ys = [1]
    sensors = np.array([])
    for y in ys:
        _sensors = np.array([load_sensor(radius, y, phi, theta) for phi in phis])
        sensors = np.append(sensors, _sensors)
    sensors = sensors.flatten()
    sensors = np.array([s.world_transform() for s in sensors])
    return sensors

def generate_sunset_taxi_cam_positions() -> np.array:
    mi.load_file("scenes/street_sunset/street_sunset.xml")
    p = mi.traverse(scene)
    cam_keys = ['PerspectiveCamera_5.to_world', \
        'PerspectiveCamera.to_world', 
        'PerspectiveCamera_1.to_world', 
        'PerspectiveCamera_2.to_world',
        'PerspectiveCamera_3.to_world',
        'PerspectiveCamera_4.to_world',         
        'PerspectiveCamera_6.to_world',
        'PerspectiveCamera_7.to_world']
    sensors = np.array([p[k] for k in cam_keys])
    return sensors

def use_provided_cam_position() -> np.array:
    #     from mitsuba import ScalarTransform4f as T  
    scene = mi.load_file("scenes/nyc_scene/nyc_scene.xml")
    p = mi.traverse(scene)
    cam_key = 'PerspectiveCamera.to_world'
    sensor = np.array([p[cam_key]])
    return sensor

def generate_cube_scene_cam_positions() -> np.array:
    """
    Load a mesh and use its vertices as camera positions
    e.g.,  Load a half-icosphere and separate the vertices by their height above target object
    each strata of vertices forms a 'ring' around the object. place cameras in a ring around the object
    and return camera positions (world_transform())
    """
    from mitsuba import ScalarTransform4f as T    
    def load_sensor_at_position(x,y,z):  
        origin = mi.ScalarPoint3f([x,y,z])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, -0.5, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
    sphere = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_mid.ply"
        },
    })
    sphere_outer = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_outer.ply"
        },
    })  
    sphere_inner = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_inner.ply"
        },
    })        
    ip = mi.traverse(sphere)
    ipv = np.array(ip["sphere.vertex_positions"])
    ipv  = np.reshape(ipv,(int(len(ipv)/3),3))    

    outer_sphere_ip = mi.traverse(sphere_outer)
    outer_sphere_ipv = np.array(outer_sphere_ip["sphere.vertex_positions"])
    outer_sphere_ipv = np.reshape(outer_sphere_ipv,(int(len(outer_sphere_ipv)/3),3))    

    inner_sphere_ip = mi.traverse(sphere_inner)
    inner_sphere_ipv = np.array(inner_sphere_ip["sphere.vertex_positions"])    
    inner_sphere_ipv = np.reshape(inner_sphere_ipv,(int(len(inner_sphere_ipv)/3),3))       
    # strata = np.array(list(set(np.round(ipv[:,1],3))))  
    # strata_2_cams =  ipv[np.where(np.round(ipv,3)[:,1] == strata[2])]    
    # strata_1_cams = ipv[np.where(np.round(ipv,3)[:,1] == strata[1])]    
    ipv_f = ipv[np.where(ipv[:,0] > 0)]
    outer_sphere_ipv_f = outer_sphere_ipv[np.where(outer_sphere_ipv[:,0] > 0)]
    inner_sphere_ipv_f = inner_sphere_ipv[np.where(inner_sphere_ipv[:,0] > 0)]
    cam_pos_ring = np.concatenate((ipv_f, outer_sphere_ipv_f, inner_sphere_ipv_f))
    positions = np.array([load_sensor_at_position(p[0], p[1], p[2]).world_transform() for p in cam_pos_ring])
    return positions

def gen_cam_positions(z,r,size) -> np.ndarray:
    """
    Generates # cam positions of length (size) in a circle of radius (r) 
    at the given latitude (z) on a sphere.  Think of the z value as the height above/below the object
    you want to render.  

    The sphere is centered at the origin (0,0,0) in the scene.  
    """
    if z > r:
        raise Exception("z value must be less than or equal to the radius of the sphere")
    lat_r = np.sqrt(r**2 - z**2)  # find latitude circle radius
    num_points = np.arange(1,size+1)
    angles = np.array([(2 * np.pi * p / size) for p in num_points])
    vertices = np.array([np.array([np.cos(a)*lat_r, z, np.sin(a)*lat_r]) for a in angles])
    return vertices

def load_sensor_at_position(x,y,z):  
    from mitsuba import ScalarTransform4f as T        
    origin = mi.ScalarPoint3f([x,y,z])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            #target=[0, -0.5, 0],
            target=[0, 0, 0],
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })

def generate_cam_positions_for_lats(lats=[], r=None, size=None, reps_per_position=1):
    """
    Wrapper function to allow generation of camera angles for any list of arbitrary latitudes
    Note that the latitudes must be some z value within the pos/neg value of the radius in the sphere:
    so: {z | -r <= z <= r}
    """
    all_pos = gen_cam_positions(lats[0], r, size)
    for i in range(1,len(lats)):
        p = gen_cam_positions(lats[i], r, size)
        all_pos = np.concatenate((all_pos, p), axis=0)
      
    positions = np.array([load_sensor_at_position(p[0], p[1], p[2]).world_transform() for p in all_pos])
    positions = np.repeat(positions, reps_per_position)
    return positions    

def generate_cube_scene_1_orbit_cam_positions(reps_per_position=1) -> np.array:
    """
    Wrapper function to generate 4 cam positions @ 3 latitutdes
    """
    # r = 14
    r = 3
    size = 4 # desired # pts on the latitude circle
    # z_lats = [8.0,10.0,12.0] # values derived from Blender
    z_lats = [1.5,2,2.5] # values derived from Blender
    positions = generate_cam_positions_for_lats(z_lats, r, size)
    return positions

def generate_cube_scene_4_orbit_cam_positions(reps_per_position=1) -> np.array:
    """
    Wrapper function to generate 4 cam positions @ 3 latitutdes
    """
    # r = 14
    r = 3
    size = 4 # desired # pts on the latitude circle
    # z_lats = [8.0,10.0,12.0] # values derived from Blender
    z_lats = [1.5,2,2.5] # values derived from Blender
    positions = generate_cam_positions_for_lats(z_lats, r, size)
    return positions

def generate_cube_scene_8_orbit_cam_positions(reps_per_position=1) -> np.array:
    """
    Wrapper function to generate 8 cam positions @ 3 latitutdes
    """
    r = 11
    size=8 # desired # pts on the latitude circle
    z_lats = [2.1381900311, 4.1942100525, 6.0890493393] # values derived from Blender
    positions = generate_cam_positions_for_lats(z_lats, r, size)
    return positions

def generate_cube_scene_16_orbit_cam_positions(reps_per_position=1) -> np.array:
    """
    Wrapper function to generate 32 cam positions @ 3 latitutdes
    """
    r = 11
    size=16 # desired # pts on the latitude circle
    z_lats = [2.1381900311, 4.1942100525, 6.0890493393] # values derived from Blender
    positions = generate_cam_positions_for_lats(z_lats, r, size)
    return positions

def generate_cube_scene_32_orbit_cam_positions(reps_per_position=1) -> np.array:
    """
    Wrapper function to generate 32 cam positions @ 3 latitutdes
    """
    r = 11
    size=32 # desired # pts on the latitude circle
    z_lats = [2.1381900311, 4.1942100525, 6.0890493393] # values derived from Blender
    positions = generate_cam_positions_for_lats(z_lats, r, size)
    return positions

def generate_cube_scene_64_orbit_cam_positions(reps_per_position=1) -> np.array:
    """
    Wrapper function to generate 64 cam positions @ 3 latitutdes
    """
    r = 11
    size=64 # desired # pts on the latitude circle
    z_lats = [2.1381900311, 4.1942100525, 6.0890493393] # values derived from Blender
    positions = generate_cam_positions_for_lats(z_lats, r, size)
    return positions

def attack_dt2(cfg:DictConfig) -> None:

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", default=1)
    DEVICE = "cuda:0"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dt2")

    batch_size = cfg.attack.batch_size
    eps = cfg.attack.eps
    eps_step =  cfg.attack.eps_step
    targeted =  cfg.attack.targeted
    target_class = cfg.attack.target_idx
    target_string = cfg.attack.target
    iters = cfg.attack.iters
    spp = cfg.attack.samples_per_pixel
    scene_file = cfg.attack.scene.path
    param_key = cfg.attack.scene.target_param_key
    sensor_key = cfg.attack.scene.sensor_key
    score_thresh = cfg.model.score_thresh_test
    weights_file = cfg.model.weights_file 
    model_config = cfg.model.config
    sensor_positions = cfg.scenario.sensor_positions.function
    randomize_sensors = cfg.scenario.randomize_positions 
    scene_file_dir = os.path.dirname(scene_file)
    tex_path = cfg.attack.scene.tex
    tmp_perturbation_path = os.path.join(f"{scene_file_dir}",f"textures/{target_string}_tex","tmp_perturbations")
    if os.path.exists(tmp_perturbation_path) == False:
        os.makedirs(tmp_perturbation_path)
    render_path = os.path.join(f"renders",f"{target_string}")
    if os.path.exists(render_path) == False:
        os.makedirs(render_path)
    preds_path = os.path.join("preds",f"{target_string}")
    if os.path.exists(preds_path) == False:
        os.makedirs(preds_path)    
    
    # img_name = 'render_39_s3'
    # adv_path = f'/nvmescratch/mhull32/robust-models-transfer/renders/{img_name}.jpg'
    mi.set_variant('cuda_ad_rgb')
    mitsuba_tex = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'filename': tex_path,
        'raw': True
    })
    mt = mi.traverse(mitsuba_tex)
    # FIXME - allow variant to be set in the configuration.
    # scene = mi.load_file("scenes/street_sunset/street_sunset.xml")    
    scene = mi.load_file(scene_file)
    p = mi.traverse(scene)    
    # 4-way intersection Taxi Texture Map
    # k = 'mat-13914_Taxi_car.001.brdf_0.base_color.data'
    # foldable car texture map
    # k = 'mat-Material.brdf_0.base_color.data'
    # sunset street taxi texture map
    # k = 'mat-Material.brdf_0.base_color.data' 
    k = param_key
    # set the texture with the bitmap from config
    p[k] = mt['data']
    # Stop Sign Texture Map
    # k = 'mat-Material.005.brdf_0.base_color.data'
    # Camera that we want to transform
    # k1 = 'PerspectiveCamera_10.to_world'
    k1 = f'{sensor_key}.to_world'
    k2 = f'{sensor_key}.film.size'
    p.keep([k,k1])
    p.update()

    # moves_matrices = generate_stop_sign_approach_cam_moves()
    # moves_matrices = generate_taxi_cam_positions()
    # moves_matrices = generate_sunset_taxi_cam_positions()
    # moves_matrices = np.tile(p[k1],1)
    # moves_matrices = generate_cube_scene_cam_positions()
    # moves_matrices = generate_cube_scene_orbit_cam_positions()
    moves_matrices = eval(sensor_positions + "()")
    if randomize_sensors:
        np.random.shuffle(moves_matrices)

    # sanity check-render the scene
    # with dr.suspend_grad():
    #     img = mi.render(scene, params = p, spp=256)
    #     mi.util.write_bitmap("renders/render.png", data=img)

    # load pre-trained robust faster-rcnn model
    dt2_config = get_cfg()
    dt2_config.merge_from_file(model_config)
    dt2_config.MODEL.WEIGHTS = weights_file
    dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    # FIXME - Get GPU Device form environment variable.
    dt2_config.MODEL.DEVICE = DEVICE
    model = build_model(dt2_config)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_config.MODEL.WEIGHTS)
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True
    # smoke test - get a pred!
    # input = dt2_input(image_path=adv_path)
    # save_adv_image_preds(model=model, input=input, path=f'dt2_prediction_img/{img_name}_preds.jpg')
    # print('')

    # rn = 10
    # s = 8
    # nput = dt2_input(f'renders/render_{rn}_s{s}.jpg')
    # save_adv_image_preds(model=model, input=nput, instance_mask_thresh=0.2, path=f'preds/render_{rn}_s{s}.jpg')


    def optim_batch(scene, batch_size, camera_positions, spp, k, label, iters, alpha, epsilon, targeted=False):
        # run attack
        if targeted:
            assert(label is not None)

        assert(batch_size <= len(camera_positions))
        success = False
        
        # wrapper function that models the input image and returns the loss
        # TODO - 2 model input should accept a batch
        @dr.wrap_ad(source='drjit', target='torch')
        def model_input(x, target):
            """
            To get the losses using DT2, we must supply the Ground Truth w/ the input dict
            as an Instances object. This includes the ground truth boxes (gt_boxes)
            and ground truth classes (gt_classes).  There should be a class & box for 
            each GT object in the scene.
            """
            losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]            
            target_loss_idx = [0] # this targets only `loss_cls` loss
            # detectron2 wants images as RGB 0-255 range
            x = ch.clip(x * 255 + 0.5, 0, 255).requires_grad_()
            x = ch.permute(x, (0, 3, 1, 2)).requires_grad_()
            x.retain_grad()
            height = x.shape[2]
            width = x.shape[3]
            instances = Instances(image_size=(height,width))
            instances.gt_classes = target.long()
            # taxi bbox
            # instances.gt_boxes = Boxes(ch.tensor([[ 50.9523, 186.4931, 437.6184, 376.7764]]))
            # stop sign bbox
            instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))
            inputs = list()
            for i in  range(0, x.shape[0]):                
                input = {}
                input['image']  = x[i]    
                input['filename'] = ''
                input['height'] = height
                input['width'] = width
                input['instances'] = instances      
                inputs.append(input)
            with EventStorage(0) as storage:            
                # loss = model([input])[losses_name[target_loss_idx[0]]].requires_grad_()
                losses = model(inputs)
                loss = sum([losses[losses_name[tgt_idx]] for tgt_idx in target_loss_idx]).requires_grad_()
            return loss

        params = mi.traverse(scene)
        if isinstance(params[k], dr.cuda.ad.TensorXf):
            # use Float if dealing with just texture colors (not a texture map)
            orig_tex = dr.cuda.ad.TensorXf(params[k])
        elif isinstance(params[k], dr.cuda.ad.Float):
            orig_tex = dr.cuda.ad.Float(params[k])        
        else:
            raise Exception("Unrecognized Differentiable Parameter Data Type.  Should be one of dr.cuda.ad.Float or dr.cuda.ad.TensorXf")

        orig_tex.set_label_("orig_tex")
        
        # indicate sensors to use in producing the perturbation
        # e.g., [0,1,2,3] will use sensors 0-3 focus on Taxi/Cement Truck in 'intersection_taxi.xml'
        # sensor 10 is focused on stop sign.
        sensors = [0]
        if iters % len(sensors) != 0:
            print("uneven amount of iterations provided for sensors! Some sensors will be used more than others\
                during attack")
        # if only one camera in the scene, then this idx will be repeated for each iter
        camera_idx = ch.Tensor(np.array(sensors)).repeat(int(iters/len(sensors))).to(dtype=ch.uint8).numpy().tolist()
        # one matrix per camera position that we want to render from, equivalent to batch size
        # e.g., batch size of 5 = 5 required camera positions
        
        cam_idx = 0
        for it in range(iters):
            # print(f'iter {it}')
            # logger.info(f"iter {it}")
            # keep 2 sets of parameters because we only need to differentiate wrt texture
            diff_params = mi.traverse(scene)
            non_diff_params = mi.traverse(scene)
            diff_params.keep(k)
            non_diff_params.keep([k1,k2])
            # optimizer is not used but necessary to instantiate to get gradients from diff rendering.
            opt = mi.ad.Adam(lr=0.1, params=diff_params)
            dr.enable_grad(orig_tex)
            dr.enable_grad(opt[k])
            opt[k].set_label_("bitmap")

            # sample random camera positions (=batch size) for each batch iteration
            if camera_positions.size > 1:
                np.random.seed(it+1)
                sampled_camera_positions_idx = np.random.randint(low=0, high=len(camera_positions)-1,size=batch_size)
            else: sampled_camera_positions_idx = [0]
            # sampled_camera_positions = camera_positions[sampled_camera_positions_idx]
            sampled_camera_positions = camera_positions
            if success:
                cam_idx += 1
                print(f"Successful pred, using camera_idx {cam_idx}")
                logger.info(f"Successful pred, using camera_idx {cam_idx}")
            N, H, W, C = batch_size, non_diff_params[k2][0], non_diff_params[k2][1], 3
            imgs = dr.empty(dr.cuda.ad.Float, N * H * W * C)

                
            for b in range(0, batch_size):

                # EOT Strategy
                # set the camera position, render & attack
                if cam_idx > len(sampled_camera_positions)-1:
                    print(f"Successfull detections on all {len(sampled_camera_positions)} positions.")
                    logger.info(f"Successfull detections on all {len(sampled_camera_positions)} positions.")
                    return
                if isinstance(sampled_camera_positions[cam_idx], mi.cuda_ad_rgb.Transform4f):
                    non_diff_params[k1].matrix = sampled_camera_positions[cam_idx].matrix
                else:
                    non_diff_params[k1].matrix = mi.cuda_ad_rgb.Matrix4f(sampled_camera_positions[cam_idx])
                non_diff_params.update()
                params.update(opt)            

                img =  mi.render(scene, params=params, spp=spp, sensor=camera_idx[it], seed=it+1)
                img.set_label_(f"image_b{it}_s{b}")
                rendered_img_path = os.path.join(render_path,f"render_b{it}_s{b}.png")
                mi.util.write_bitmap(rendered_img_path, data=img)
                img = dr.ravel(img)
                # dr.disable_grad(img)
                start_index = b * (H * W * C)
                end_index = (b+1) * (H * W * C)
                index = dr.arange(dr.cuda.ad.UInt, start_index, end_index)                
                dr.scatter(imgs, img, index)
            
                time.sleep(1.0)
                # Get and Vizualize DT2 Predictions from rendered image
                rendered_img_input = dt2_input(rendered_img_path)
                success = save_adv_image_preds(model \
                    , dt2_config, input=rendered_img_input \
                    , instance_mask_thresh=dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST \
                    , target = label
                    , path=os.path.join(preds_path,f'render_b{it}_s{b}.png'))
                target = dr.cuda.ad.TensorXf([label], shape=(1,))

            imgs = dr.cuda.ad.TensorXf(dr.cuda.ad.Float(imgs),shape=(N, H, H, C))
            if (dr.grad_enabled(imgs)==False):
                dr.enable_grad(imgs)
            loss = model_input(imgs, target)
            sensor_loss = f"[PASS {cfg.sysconfig.pass_idx}] iter: {it} sensor pos: {cam_idx}/{len(sampled_camera_positions)}, loss: {str(loss.array[0])[0:7]}"
            print(sensor_loss)
            logger.info(sensor_loss)
            # model.train = False
            dr.enable_grad(loss)
            dr.backward(loss)

            #########################################################################
            # L-INFattack
            # grad = dr.grad(opt[k])
            # tex = opt[k]
            # eta = alpha * dr.sign(grad)
            # if targeted:
            #     eta = -eta
            # tex = tex + eta
            # eta = dr.clamp(tex - orig_tex, -epsilon, epsilon)
            # tex = orig_tex + eta
            # tex = dr.clamp(tex, 0, 1)
            #########################################################################
            HH, WW  = dr.shape(dr.grad(opt[k]))[0], dr.shape(dr.grad(opt[k]))[1]
            grad = ch.Tensor(dr.grad(opt[k]).array).view((HH, WW, C))
            tex = ch.Tensor(opt[k].array).view((HH, WW, C))
            _orig_tex = ch.Tensor(orig_tex.array).view((HH, WW, C))
            l = len(grad.shape) -  1
            g_norm = ch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
            scaled_grad = grad / (g_norm  + 1e-10)
            if targeted:
                scaled_grad = -scaled_grad
            # step
            tex = tex + scaled_grad * alpha
            delta  = tex - _orig_tex
            # project
            delta =  delta.renorm(p=2, dim=0, maxnorm=epsilon)
            tex = _orig_tex + delta

            # convert back to mitsuba dtypes            
            tex = dr.cuda.ad.TensorXf(tex.to(DEVICE))
            tex = dr.clamp(tex, 0, 1)
            
            params[k] = tex     
            dr.enable_grad(params[k])
            params.update()
            perturbed_tex = mi.Bitmap(params[k])
            
            
            mi.util.write_bitmap(os.path.join(tmp_perturbation_path,f"perturbed_tex_map_b{it}.png"), data=perturbed_tex)
            time.sleep(0.2)
            if it==(iters-1) and isinstance(params[k], dr.cuda.ad.TensorXf):
                perturbed_tex = mi.Bitmap(params[k])
                mi.util.write_bitmap("perturbed_tex_map.png", data=perturbed_tex)
                time.sleep(0.2) 
        return scene
    
    # iters = iters  
    samples_per_pixel = spp
    epsilon = eps
    alpha = eps_step #(epsilon / (iters/50))
    label = target_class
    img = optim_batch(scene, batch_size=1, camera_positions =  moves_matrices, spp=samples_per_pixel, k=k, label = label, iters=iters, alpha=alpha, epsilon=epsilon, targeted=True)

    def optim(scene, spp, k, label, iters, alpha, epsilon, targeted=False):
        # run attack
        if targeted:
            assert(label is not None)
        
        # wrapper function that models the input image and returns the loss
        @dr.wrap_ad(source='drjit', target='torch')
        def model_input(x, target):
            """
            To get the losses using DT2, we must supply the Ground Truth w/ the input dict
            as an Instances object. This includes the ground truth boxes (gt_boxes)
            and ground truth classes (gt_classes).  There should be a class & box for 
            each GT object in the scene.
            """
            input = {}
            losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]            
            target_loss_idx = [0] # this targets only `loss_cls` loss
            # detectron2 wants images as RGB 0-255 range
            x = ch.clip(x * 255 + 0.5, 0, 255).requires_grad_()
            x = ch.permute(x, (2,0,1)).requires_grad_()
            x.retain_grad()
            height = x.shape[1]
            width = x.shape[2]
            instances = Instances(image_size=(height,width))
            instances.gt_classes = target.long()
            # taxi bbox
            # instances.gt_boxes = Boxes(ch.tensor([[ 50.9523, 186.4931, 437.6184, 376.7764]]))
            # stop sign bbox
            instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))
            input['image']  = x    
            input['filename'] = ''
            input['height'] = height
            input['width'] = width
            input['instances'] = instances            
            with EventStorage(0) as storage:            
                # loss = model([input])[losses_name[target_loss_idx[0]]].requires_grad_()
                losses = model([input])
                loss = sum([losses[losses_name[tgt_idx]] for tgt_idx in target_loss_idx]).requires_grad_()
            return loss

        params = mi.traverse(scene)
        if isinstance(params[k], dr.cuda.ad.TensorXf):
            # use Float if dealing with just texture colors (not a texture map)
            orig_tex = dr.cuda.ad.TensorXf(params[k])
        elif isinstance(params[k], dr.cuda.ad.Float):
            orig_tex = dr.cuda.ad.Float(params[k])        
        else:
            raise Exception("Unrecognized Differentiable Parameter Data Type.  Should be one of dr.cuda.ad.Float or dr.cuda.ad.TensorXf")

        orig_tex.set_label_("orig_tex")
        
        # indicate sensors to use in producing the perturbation
        # e.g., [0,1,2,3] will use sensors 0-3 focus on Taxi/Cement Truck in 'intersection_taxi.xml'
        # sensor 10 is focused on stop sign.
        sensors = [0]
        if iters % len(sensors) != 0:
            print("uneven amount of iterations provided for sensors! Some sensors will be used more than others\
                during attack")
        camera_idx = ch.Tensor(np.array(sensors)).repeat(int(iters/len(sensors))).to(dtype=ch.uint8).numpy().tolist()
        # repeat the camera position matrices for the nubmer of desired renders per camera position. 
        # e.g., 50 iters / 10 sampled positions  = 5 renders per position
        camera_positions = np.tile(moves_matrices, int(iters/len(moves_matrices)))
        
        for it in range(iters):
            print(f'iter {it}')
            # keep 2 sets of parameters because we only need to differentiate wrt texture
            diff_params = mi.traverse(scene)
            non_diff_params = mi.traverse(scene)
            diff_params.keep(k)
            non_diff_params.keep(k1)
            # optimizer is not used but necessary to instantiate to get gradients from diff rendering.
            opt = mi.ad.Adam(lr=0.1, params=diff_params)
            dr.enable_grad(orig_tex)
            dr.enable_grad(opt[k])
            opt[k].set_label_("bitmap")
            
            # EOT Strategy
            # set the camera position to sampled position, render & attack
            if isinstance(camera_positions[it], mi.cuda_ad_rgb.Transform4f):
                non_diff_params[k1].matrix = camera_positions[it].matrix
            else:
                non_diff_params[k1].matrix = mi.cuda_ad_rgb.Matrix4f(camera_positions[it])
            non_diff_params.update()
            params.update(opt)            

            # TODO  1 - render from each camera viewport
            img =  mi.render(scene, params=params, spp=spp, sensor=camera_idx[it], seed=it+1)
            img.set_label_("image")
            dr.enable_grad(img)
            rendered_img_path = f"renders/render_{it}_s{camera_idx[it]}.png"
            mi.util.write_bitmap(rendered_img_path, data=img)
            time.sleep(1.0)
            # Get and Vizualize DT2 Predictions from rendered image
            rendered_img_input = dt2_input(rendered_img_path)
            success = save_adv_image_preds(model, dt2_config, rendered_img_input, path=f'preds/render_{it}_s{camera_idx[it]}.png')
            target = dr.cuda.ad.TensorXf([label], shape=(1,))
            loss = model_input(img, target)
            print(f"sensor: {str(camera_idx[it])}, loss: {str(loss.array[0])[0:7]}")
            # model.train = False
            dr.enable_grad(loss)
            dr.backward(loss)
            
            grad = dr.grad(opt[k])
            tex = opt[k]
            eta = alpha * dr.sign(grad)
            if targeted:
                eta = -eta
            tex = tex + eta
            eta = dr.clamp(tex - orig_tex, -epsilon, epsilon)
            tex = orig_tex + eta
            tex = dr.clamp(tex, 0, 1)
            params[k] = tex
            dr.enable_grad(params[k])
            params.update()
            if it==(iters-1) and isinstance(params[k], dr.cuda.ad.TensorXf):
                perturbed_tex = mi.Bitmap(params[k])
                mi.util.write_bitmap("perturbed_tex_map.png", data=perturbed_tex)
                time.sleep(0.2)
        return scene