"""
Example Usage:python src/render_batch.py 
    -s scenes/cube_scene/cube_scene.xml 
    -cm generate_taxi_cam_positions
"""

import mitsuba as mi
import drjit as dr
import os
import argparse
import time
import ast
from tqdm import tqdm

from dt2 import generate_cam_positions_for_lats, generate_batch_sensor

if __name__  == "__main__":
    parser = argparse.ArgumentParser( \
        description='Example script with default values' \
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-s", "--scene", help="Mitsuba scene file path.", required=True)
    parser.add_argument("-rx", "--sensor-width-resolution", type=int, help="resolution for sensor width")
    parser.add_argument("-ry", "--sensor-height-resolution", type=int, help="resolution for sensor height")
    parser.add_argument("-sr", "--sensor-radius", type=float, help="sensor radius")
    parser.add_argument("-sc", "--sensor-count", type=int, help="sensor count")
    parser.add_argument("-sz", "--sensor-z-lats", type=ast.literal_eval, help="sensor z lats")
    parser.add_argument("-sp", "--spp", type=int, default=256, help="samples per pixel per render")
    parser.add_argument("-od", "--outdir", help="directory for rendered images", default="renders")
    parser.add_argument("-ck", "--cam-key", help="Mitsuba Scene Params Camera Key", default='PerspectiveCamera.to_world')

    args = parser.parse_args()

    sensor_z_lats = args.sensor_z_lats    
    
    W = args.sensor_width_resolution
    H = args.sensor_height_resolution
    spp = args.spp
    
    mi.set_variant("cuda_ad_rgb")
    scene = mi.load_file(args.scene)
    
    camera_positions = generate_cam_positions_for_lats(lats=sensor_z_lats, r=args.sensor_radius, size=args.sensor_count, world_transformed=False)
    
    # idea is to "batch the batch" because > 30 sensors causes out of memory error.
    # first idea, is if batch is >30, then divide number of sensors by 30 for each batch
    # if the last batch is < 30, then create a batch of < 30 sensors
    # e.g., if batch size is 100, then 4 batches of 30,30,30,10 sensors
    # camera_positions contains the number of camera positions that we need to generate for.
    # first, divide them up into batches of 30  (or less)
    
    batched_camera_positions = []
    if len(camera_positions) > 30:
        batch_size = 30
        num_batches = len(camera_positions) // 30
        last_batch_size = len(camera_positions) % 30
        if last_batch_size > 0:
            num_batches += 1
        print(f'num_batches={num_batches}')
        for i in range(0, num_batches):
            if i == num_batches - 1:
                batched_camera_positions.append(camera_positions[i*batch_size:])
            else:
                batched_camera_positions.append(camera_positions[i*batch_size:(i+1)*batch_size])
     
     # for every batch in batched_camera_positions, generate a batch sensor
    batch_sensors = []
    num_batches = len(batched_camera_positions)
    for i, b in enumerate(batched_camera_positions):
        print(f'generating batch sensor for batch {i+1} of {num_batches}')
        batch_sensor_dict = generate_batch_sensor(b, W, H, args.spp)
        batch_sensor = mi.load_dict(batch_sensor_dict)
        batch_sensors.append(batch_sensor)
        
    params = mi.traverse(scene)

    cam_key = args.cam_key
    print(f'rendering {len(camera_positions)} imgs...')
    for i in tqdm(range(0, len(batch_sensors)), desc='Rendering Batched Images'):
        batch_sensor = batch_sensors[i]
        num_sensors = batch_sensor.m_film.size()[0]//W
        img =  mi.render(scene, params=params, spp=spp, sensor=batch_sensor, seed=i+1)
        for j in range(num_sensors):
            start = j * W
            end = (j+1) * W
            split_img = img[:,start:end,:]
            img_counter = i * batch_size + j
            rendered_img_path = os.path.join(args.outdir,f"render_{img_counter}.png")
            mi.util.write_bitmap(rendered_img_path, data=split_img, write_async=False)   
    
    print('Finished rendering')