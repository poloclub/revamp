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

from dt2 import (generate_sunset_taxi_cam_positions
,generate_cube_scene_16_orbit_cam_positions
,generate_cube_scene_32_orbit_cam_positions
,generate_cube_scene_64_orbit_cam_positions
,generate_stop_sign_approach_cam_moves
,generate_taxi_cam_positions
,generate_cube_scene_cam_positions)

if __name__  == "__main__":
    parser = argparse.ArgumentParser( \
        description='Example script with default values' \
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-s", "--scene", help="Mitsuba scene file path.", required=True)
    parser.add_argument("-cm", "--cams", help="Function generating camera position matrices", required=True)
    parser.add_argument("-sp", "--spp", type=int, default=256, help="samples per pixel per render")
    parser.add_argument("-od", "--outdir", help="directory for rendered images", default="renders")
    parser.add_argument("-ck", "--cam-key", help="Mitsuba Scene Params Camera Key", default='PerspectiveCamera.to_world')

    args = parser.parse_args()

    mi.set_variant("cuda_ad_rgb")
    scene = mi.load_file(args.scene)
    
    camera_positions = eval(args.cams + "()")
    params = mi.traverse(scene)
    cam_key = args.cam_key
    print(f'rendering {len(camera_positions)} imgs...')
    for i in range(0, len(camera_positions)):
        params[cam_key].matrix = camera_positions[i].matrix
        params.update()
        img =  mi.render(scene, params=params, spp=256, sensor=0, seed=i+1)
        rendered_img_path = os.path.join(args.outdir,f"render_{i}.png")
        mi.util.write_bitmap(rendered_img_path, data=img)
        time.sleep(0.2)
    print('done')