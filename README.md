# Texture Attacks using Differentiable Rendering

## What does this project do?  
Generate physically realizable, robust adversarial textures for 3D objects using photorealistic differentiable rendering. 

## Motivation
Generating adversarial examples in the image space has been widely studied.  However, limited progress has been made toward generating physically realizable adversarial examples where an adversary is constrained to only perturbing a few parameters, such as texture or lighting.  Differentiable rendering permits study of these types of attacks digitally using a photorealistic process.  

## What is the threat model?
The attacker executes a white-box (PGD L2 / Linf) perturbation attack constrained to the texture of an object rendered in a 3D scene that fools an Image Classifer or Object Detector into detecting the target class. 

The objective is to find a texture perturbation that is consistently classified / detected as the target class over many transformations of the scene parameters. _i.e.,_ sensor position and lighting. 

## How is differentiable rendering used?

A differentiable renderer allows optimization of the underlying 3D scene parameters by obtaining useful gradients of the rendering process. In this project, a rendered image of a scene is passed to the victim model (image classifier / object detector). Next, the model's loss is backpropogated through the differentiable renderer to the scene parameters, _e.g._, object texture, object vertex positions, lighting, _etc._. Finally, the chosen scene parameter is iteratively perturbed to fool the model and the scene is re-rendered until the attack succeeds. 

## How is this project structured?
This project uses configurable scenarios that can be used to create experiments for a variety of studies.  At the highest level, a scenario generally consists of a 3D scene, an attackable parameter, an attack success metric, and a victim model.  

For example, one scenario uses a "cube scene" consisting of a single cube and some lights.  The attackable parameter is the cube texture. 

`make scenario configs`

See `Makefile` for available commands

### Examples
Run a texture attack on Detectron2 and log the results to a file

`python src/run.py attack.target=cat sysconfig.output_path=results/cat`

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

`make scenes/cube_scene/textures/traffic_light_tex/tex_2.png.unset_tex`
