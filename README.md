# Differentiable Rendering Texture Attack

See `Makefile` for available commands

### Examples
Run a texture attack on Detectron2 and log the results to a file

`make attack_dt2 > results/results.txt`

### Clean-up renders/ predicted renders images

`make clean`


## Pipeline to use an attacked texture, render a batch of images, and predict results. 
Here `TEX_NUM=0` refers to the index of the texture

`make TEX_NUM=0 render_predict` 

Other params
`TARGET_SCENE = cube_scene
TARGET_TEX  = traffic_light_tex
ORIG_TEX = red_tex.png 
TEX_NUM = 0`

e.g., Use the file `tex_0.png` for the `PERSON` class and render/predict a batch of images.
`make TARGET_TEX = person TEX_NUM=0 render_predict`


### Proccess output logs into loss results - outputs `{filename}.csv`

`python src/results.py -i results/results.txt`

### Process output scores into score results - outputs `{filename}.csv`

`python src/scores.py -i results/person/0_scores.txt -t person`

### Set an (optional alternate) texture before rendering

`make scenes/cube_scene_c/textures/traffic_light_tex/tex_2.png.set_tex`

or 

`make TARGET_TEX=stop_sign_tex scenes/cube_scene/textures/stop_sign_tex/tex_0.png.set_tex`

### Render batch of images
This command generates renders from 48 sensor positions. See `dt2.py` for details.

Other args to `-cm` 
This generates 264 sensor positions at vertices of 3 concentric half-icospheres

`generate_cube_scene_orbit_cam_positions` 

generate_cube_scene_cam_positions

### Prep batch for detection

`make img_to_pred`

### Predict on a batch of images

`python src/predict_objdet_batch.py -d red_cube -st 0.3 > results/tf0_scores.txt`

### Unset a texture and use original tex

`make scenes/cube_scene_c/textures/traffic_light_tex/tex_2.png.unset_tex`
