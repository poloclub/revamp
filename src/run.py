import argparse
import json
import os
import subprocess
import shutil
import glob
import numpy as np
from detectron2.data import MetadataCatalog
import dt2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config" \
        ,type=str \
        ,default = "scenario_configs/coco_sports_ball.json" \
        ,help="Path to the JSON-formatted config file")
    args = parser.parse_args()

    config_path = args.config
    original_cwd = os.getcwd()

    with open(config_path, "r") as schema_file:
        schema = json.load(schema_file)
        passes = schema["attack"]["kwargs"]["passes"]
        passes_names = schema["attack"]["kwargs"]["passes_names"]
        batch_size = schema["attack"]["kwargs"]["batch_size"]
        eps = schema["attack"]["kwargs"]["eps"]
        eps_step = schema["attack"]["kwargs"]["eps_step"]
        targeted = schema["attack"]["kwargs"]["targeted"]
        target = schema["attack"]["kwargs"]["target"]
        iters = schema["attack"]["kwargs"]["iters"]
        spp = schema["attack"]["kwargs"]["samples_per_pixel"]
        scene_file = schema["attack"]["kwargs"]["scene_file"]
        target_param_key = schema["attack"]["kwargs"]["target_param_key"]
        sensor_key = schema["attack"]["kwargs"]["sensor_key"]

        score_thresh_test = schema["model"]["model_kwargs"]["score_thresh_test"]
        weights_file = schema["model"]["weights_file"]
        model_config = schema["model"]["config"]

        library = schema["dataset"]["library"]
        dataset = schema["dataset"]["name"]
        
        # parse the target class string and determine index
        # e.g., person class is @ index 0 in coco_train_2017
        if library == "detectron2":
            # TODO - raise exception if target class is not found in DT2
            # handle 2-word classes e.g., : "sports_ball" --> "sports ball"
            target = target.lower()
            formatted_target = target.replace("_", " ")
            classes = MetadataCatalog.get(dataset).thing_classes
            target_index = classes.index(formatted_target)

        sensor_positions = schema["scenario"]["scenario_kwargs"]["sensor_positions"]
        randomize_sensors = schema["scenario"]["scenario_kwargs"]["randomize_sensors"]

        output_path = schema["sysconfig"]["output_path"]
        if os.path.exists(output_path) == False:
            os.mkdir(output_path)
        

    opts = [
        " -bs {}".format(batch_size),
        " -e {}".format(eps),
        " -es {}".format(eps_step),
        " -t {}".format(targeted),
        " -tc {}".format(target_index),
        " -ts {}".format(target),
        " -it {}".format(iters),
        " -sp {}".format(spp),
        " -sf {}".format(scene_file),
        " -pk {}".format(target_param_key),
        " -sk {}".format(sensor_key),
        " -st {}".format(score_thresh_test),
        " -wf {}".format(weights_file),
        " -mc {}".format(model_config),
        " -p {}".format(sensor_positions),
        " -rs {}".format(randomize_sensors),
        " > {}".format(output_path)
    ]

    passes = list(range(passes))
    if passes_names is not None:
        passes = [int(p) for p in passes_names]


    for i in range(len(passes)):
        fn = f"{passes[i]}.txt"
        opts[-1] = f"> {os.path.join(output_path, fn)}"
        command = "python src/dt2.py" + " ".join(opts)
        print(command)        
        subprocess.run(command, shell=True, check=True)

        # copy last texture perturbation to use for next perturbation
        tex_dir = os.path.join(os.path.dirname(scene_file), "textures", f"{target}_tex")
        tmp_dir = "tmp_perturbations"
        texs = os.listdir(os.path.join(tex_dir, tmp_dir))
        os.chdir(os.path.join(tex_dir, tmp_dir))
        # get most recent timestamped perturbation
        last_tex = max(texs, key=lambda x: os.path.getmtime(x))
        os.chdir(original_cwd)
        shutil.copy(os.path.join(tex_dir,tmp_dir, last_tex),os.path.join(tex_dir, f"tex_{passes[i]}.png"))
        os.chdir(os.path.join(tex_dir, tmp_dir))
        # rm tmp perturbations
        png_files = glob.glob("*.png")
        for file in png_files:
            os.remove(file)
        os.chdir(original_cwd)
        
        # make predictions using the same camera angles utilized for producing perturbation
        # FIXME - ensure camera position is same as used in config! 
        render_predict = f"make TARGET={target} TEX_NUM={passes[i]} render_predict"
        subprocess.run(render_predict, shell=True, check=True)

        next_tex = os.path.join(tex_dir, f"tex_{passes[i]}.png")
        set_tex = f"make TARGET={target} {next_tex}.set_tex"
        subprocess.run(set_tex, shell=True, check=True)
