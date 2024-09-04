import os, PIL, csv
import numpy as np
import torch as ch
from torchvision.io import read_image
import mitsuba as mi
import drjit as dr
import warnings
from omegaconf import DictConfig
import logging
from tqdm import tqdm
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.structures import Boxes, Instances
from detectron2.utils.events import EventStorage
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
from detectron2.data.detection_utils import *


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
    # TODO - review this class setting... why is it needed?
    instances.gt_classes = ch.Tensor([2])
    # taxi bbox
    # instances.gt_boxes = Boxes(ch.tensor([[ 50.9523, 186.4931, 437.6184, 376.7764]]))
    # stop sign bbox
    # instances.gt_boxes = Boxes(ch.tensor([[ 162.0, 145.0, 364.0, 324.0]])) # for 512x512 img
    # instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, height, width]]))
    instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))
    input['image'] = adv_image_tensor    
    input['filename'] = filename
    input['height'] = height
    input['width'] = width
    input['instances'] = instances
    return input

def save_adv_image_preds(model \
    , dt2_config \
    , input \
    , instance_mask_thresh=0.7 \
    , target:int=None \
    , untarget:int=None
    , is_targeted:bool=True \
    , format="RGB" \
    , path:str=None):
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
        untarget_pred_not_exists = untarget not in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True  
    PIL.Image.fromarray(pred).save(path)
    if is_targeted and target_pred_exists:
        return True
    elif (not is_targeted) and (untarget_pred_not_exists):
        return True
    return False

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

def load_sensor_at_position(x,y,z, resx=None, resy=None, spp=None):  
    from mitsuba import ScalarTransform4f as T        
    origin = mi.ScalarPoint3f([x,y,z])

    return mi.load_dict({
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 1, 0]
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
    
    
    
    for i,p in enumerate(camera_positions):
        
        target_x = 0
        target_y = 0
        target_z = 0
        up_vec_y = 1
        up_vec_z = 0

        if randomize_sensors:
            target_x = np.random.uniform(-0.2,0.2)
            target_y = np.random.uniform(-0.2,0.2)
            target_z = np.random.uniform(-0.2,0.2)
            up_vec_z = np.random.uniform(-1,1)
            up_vec_y = 1- np.abs(up_vec_z)
        
        origin = mi.ScalarPoint3f([p[0], p[1], p[2]])
        batch_sensor[f'sensor{i}'] = {
                'type': 'perspective',
                'fov': 39.3077,
                'to_world': T.look_at(
                    origin=origin,
                    target=[target_x, target_y, target_z], # defaults is [0,0,0]
                    up=[0, up_vec_y, up_vec_z] # defaults is [0,1,0]
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

         

def attack_dt2(cfg:DictConfig) -> None:

    cuda_visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", default=1)
    DEVICE = f"cuda:{cuda_visible_device}"

    logging.basicConfig(level=logging.INFO)
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
    multi_pass_rendering = cfg.attack.multi_pass_rendering
    multi_pass_spp_divisor = cfg.attack.multi_pass_spp_divisor
    scene_file = cfg.scene.path
    param_keys = cfg.scene.target_param_keys
    sensor_key = cfg.scene.sensor_key
    score_thresh = cfg.model.score_thresh_test
    weights_file = cfg.model.weights_file 
    model_config = cfg.model.config
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
        # refactor gen cam positions to return both world transformed and non transformed as a tuple. 
        cam_position_matrices, world_transformed_cam_position_matrices = generate_cam_positions_for_lats(lats=cfg.scene.sensor_z_lats \
                                                            ,r=cfg.scene.sensor_radius \
                                                            ,size=cfg.scene.sensor_count \
                                                            ,resx=resx \
                                                            ,resy=resy \
                                                            ,spp=spp)
        # then assign the needed one to moves_matrices
        # in either case, we will use the batch sensor to render the b&w images for calculating the
        # bboxes 
        if use_batch_sensor: 
            moves_matrices =  cam_position_matrices
            batch_sensor_dict = generate_batch_sensor(moves_matrices, resx, resy, spp)
            batch_sensor = mi.load_dict(batch_sensor_dict)
            sensor_count =  batch_sensor.m_film.size()[0]//resx #divide by resx to get number of sensors
        else: 
            moves_matrices =  world_transformed_cam_position_matrices       
            if randomize_sensors:
                np.random.shuffle(moves_matrices)
    
    #FIXME - truncate some of the camera positions, when we don't want to render an entire orbit
    # moves_matrices = moves_matrices[10:]
    # reverse moves_matrices
    # moves_matrices = moves_matrices[::-1] 
    # concat moves_matrices with moves_matrices[24:]
    # moves_matrices = np.concatenate((moves_matrices[0:5], moves_matrices[25:][::-1]), axis=0)

    # load pre-trained robust faster-rcnn model
    dt2_config = get_cfg()
    dt2_config.merge_from_file(model_config)
    dt2_config.MODEL.WEIGHTS = weights_file
    dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    dt2_config.MODEL.DEVICE = DEVICE
    model = build_model(dt2_config)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_config.MODEL.WEIGHTS)
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True

    def optim_batch(scene, batch_size, camera_positions, spp, k, label, unlabel, iters, alpha, epsilon, targeted=False):
        # run attack
        if targeted:
            assert(label is not None)

        #assert(batch_size <= sensor_count)
        assert(batch_size <= camera_positions.size)
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
            target_loss_idx = [0] # this targets es only `loss_cls` loss
            # detectron2 wants images as RGB 0-255 range
            x = ch.clip(x * 255 + 0.5, 0, 255).requires_grad_()
            x = ch.permute(x, (0, 3, 1, 2)).requires_grad_()
            x.retain_grad()
            height = x.shape[2]
            width = x.shape[3]
            instances = Instances(image_size=(height,width))
            instances.gt_classes = target.long()
            #FIXME - need better way to calculate bboxes.
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
            del x
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

            orig_tex.set_label_(f"{k}_orig_tex")
            orig_texs.append(orig_tex)
        
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
                opt[k].set_label_(f"{k}_bitmap")
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
                print(f"Successful pred, using camera_idx {cam_idx}")
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
                        print(f"Successfull detections on all {len(sampled_camera_positions)} positions.")
                        logger.info(f"Successfull detections on all {len(sampled_camera_positions)} positions.")
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

                if multi_pass_rendering:
                    # achieve the affect of rendering at a high sample-per-pixel (spp) value 
                    # by rendering multiple times at a lower spp and averaging the results
                    mini_pass_spp = spp//multi_pass_spp_divisor
                    render_passes = mini_pass_spp
                    
                    if render_passes * H * W * C > 2**32:
                        # depending on rendering configs, max arr size is 2^32=4294967296 entries                    
                        warnings.warn("The product of render_passes, H, W, and C is greater than or equal to 2^32")
                    mini_pass_renders = dr.empty(dr.cuda.ad.Float, render_passes * H * W * C)
                    for i in range(render_passes):
                        seed = np.random.randint(0,1000)+i
                        img_i = mi.render(scene, params=params, spp=mini_pass_spp, sensor=sensor_i, seed=seed)
                        s_index = i * (H * W * C)
                        e_index = (i+1) * (H * W * C)
                        mini_pass_index = dr.arange(dr.cuda.ad.UInt, s_index, e_index)
                        img_i = dr.ravel(img_i)
                        dr.scatter(mini_pass_renders, img_i, mini_pass_index)
                        
                    @dr.wrap_ad(source='drjit', target='torch')
                    def stack_imgs(imgs):
                        # drjit cannot calculate the channel-wise mean of a 4D tensor!
                        imgs = ch.mean(imgs,axis=0)
                        return imgs

                    # mini_pass_renders = dr.cuda.ad.TensorXf(mini_pass_renders, dr.shape(mini_pass_renders))
                    mini_pass_renders = dr.cuda.ad.TensorXf(mini_pass_renders, shape=(render_passes, H, W, C))
                    img = stack_imgs(mini_pass_renders)
                else: # dont use multi-pass rendering
                    img = mi.render(scene, params=params, spp=spp, sensor=sensor_i, seed=it+1)
                
                # TODO - place this into a utils function
                # split image into images for number of sensors if needed!
                # split_imgs = []
                # for i in range(len(camera_positions)):
                #     start = i * H
                #     end = (i+1) * H
                #     split_img = img[:,start:end,:]
                #     # mi.util.write_bitmap(f"split_img_{i}.png", data=split_img, write_async=False)   
                #     split_imgs.append(dr.ravel(split_img))
                    
                img.set_label_(f"image_b{it}_s{b}")
                rendered_img_path = os.path.join(render_path,f"render_b{it}_p{b}_s{cam_idx}.png")
                mi.util.write_bitmap(rendered_img_path, data=img, write_async=False)
                img = dr.ravel(img)
     
                # Get and Vizualize DT2 Predictions from rendered image
                rendered_img_input = dt2_input(rendered_img_path)
                success = save_adv_image_preds(model \
                    , dt2_config, input=rendered_img_input \
                    , instance_mask_thresh=dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST \
                    , target = label
                    , untarget = unlabel
                    , is_targeted = targeted
                    , path=os.path.join(preds_path,f'render_b{it}_s{cam_idx}.png'))
                target = dr.cuda.ad.TensorXf([label], shape=(1,))

            imgs = dr.cuda.ad.TensorXf(dr.cuda.ad.Float(img),shape=(N, H, W//N, C))
            if (dr.grad_enabled(imgs)==False):
                dr.enable_grad(imgs)
            loss = model_input(imgs, target)
            sensor_loss = f"[PASS {cfg.sysconfig.pass_idx}] iter: {it} sensor pos: {cam_idx}/{len(sampled_camera_positions)}, loss: {str(loss.array[0])[0:7]}"
            print(sensor_loss)
            logger.info(sensor_loss)
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
            for i, k in enumerate(param_keys):
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
                tex = dr.clamp(tex, 0, 1)
                params[k] = tex     
                dr.enable_grad(params[k])
                params.update()
                perturbed_tex = mi.Bitmap(params[k])
                                
                mi.util.write_bitmap(os.path.join(tmp_perturbation_path,f"{k}_{it}.png"), data=perturbed_tex, write_async=False)
                if it==(iters-1) and isinstance(params[k], dr.cuda.ad.TensorXf):
                    perturbed_tex = mi.Bitmap(params[k])
                    mi.util.write_bitmap("perturbed_tex_map.png", data=perturbed_tex, write_async=False)
                ch.cuda.empty_cache()
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
    