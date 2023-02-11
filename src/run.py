import argparse
import json
import os
import subprocess
import shutil
import glob
import numpy as np
from detectron2.data import MetadataCatalog
from dt2 import attack_dt2
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    original_cwd = os.getcwd()
    passes = cfg.attack.passes
    passes_names = cfg.attack.passes_names
    target = cfg.attack.target
    scene_file = cfg.attack.scene.path

    dataset = cfg.dataset.name 
    library = cfg.dataset.library

    if library == "detectron2":
        # TODO - raise exception if target class is not found in DT2
        # handle 2-word classes e.g., : "sports_ball" --> "sports ball"
        target = target.lower()
        cfg.attack.target = target
        formatted_target = target.replace("_", " ")
        classes = MetadataCatalog.get(dataset).thing_classes
        target_index = classes.index(formatted_target)
        cfg.attack.target_idx = target_index

    output_path = cfg.sysconfig.output_path
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)

    passes = list(range(passes))
    if passes_names is not None:
        passes = [int(p) for p in passes_names]

    for i in range(len(passes)):
        fn = f"{passes[i]}.txt"
        cfg.sysconfig.output_path = os.path.join(cfg.sysconfig.output_path, fn)
        cfg.sysconfig.pass_idx = passes[i]
        attack_dt2(cfg)

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
        render_predict = f"make TARGET={target} RESULTS_DIR={cfg.sysconfig.log_dir} TEX_NUM={passes[i]} render_predict"
        subprocess.run(render_predict, shell=True, check=True)

        next_tex = os.path.join(tex_dir, f"tex_{passes[i]}.png")
        set_tex = f"make TARGET={target} {next_tex}.set_tex"
        subprocess.run(set_tex, shell=True, check=True)

    # process logfile
    process_logs = f"python src/results.py -i {cfg.sysconfig.log_dir}/run.log"
    subprocess.run(process_logs, shell=True, check=True)

if __name__ == "__main__":
    run()