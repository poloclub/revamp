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

## How should this project be used?
This project uses configurable scenarios that can be used to create experiments for a variety of studies.  At the highest level, a scenario generally consists of a 3D scene, an attackable parameter, render settings, and a victim model.  

For example, one scenario uses a "cube scene" consisting of a single cube mesh and some lights.  The attackable parameter is the cube's texture in bitmap format. The victim model is a 2-stage object detector (faster-rcnn).  The rendering settings specify that the scene be rendered 48 different sensor positions during the attack.  


## Getting Started

`conda env create -f environment.yml`

Install Detectron2

`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

### Model Weights
We use [Robust ImageNet Models](https://github.com/microsoft/robust-models-transfer). You'll need to choose an appropriate model for your experiment. Currently we use this ResNet-50 [L2-Robust Model](https://robustnessws4285631339.blob.core.windows.net/public-models/robust_imagenet/resnet50_l2_eps0.03.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D) for object detection with Detectron2. You can specify the weights file path and model config path in `configs/model/{model}.yaml`.


### Examples
Run a texture attack on Detectron2 and log the results to a file.  We use Hydra for configuring experiments and you can easily supply your own Hydra-style config arguments. See this [Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)

#### Specify Target Class and Camera Positioning
`CUDA_VISIBLE_DEVICES=0 python src/revamp.py attack.target=cat scenario.randomize_positions=true`

#### Specify Target Class and Use a Different Scene
`CUDA_VISIBLE_DEVICES=0 python src/revamp.py attack.target=cat attack.passes=1 scene=cube_r1`

#### Resuming Experiments and Handling Out-of-Memory
Sometimes Mitsuba crashes or you want to add additional passes to an perturbed texture.  To resume / add on to an experiment, follow these steps:
Continue an experiment by adding extra passes with explicit pass names.
1. Set the texture:
` make TARGET=truck TARGET_SCENE=cube scenes/cube/textures/truck_tex/tex_6.png.set_tex`
2. run: `python src/revamp.py attack.target=truck attack.passes=1 attack.passes_names=[7]`


Rendering large scenes that contain a high number of meshes, it is possible to exhaust the memory of the GPU, resulting in out-of-memory errors.  To solve this problem, you can choose to render at a lower number of samples per pixel (SPP) or lower the resolution of the sensor.  Alternatively, you can render a sequence of images at a lower SPP value while varying the seed of each render and then average the resulting images to compose a lower noise image.  The following command uses `attack.multi_pass_rendering=true`, `attack.samples_per_pixel=64`, and `attack.multi_pass_spp_divisor=8` to use render `64/8=8` images and then averages them together.

`CUDA_VISIBLE_DEVICES=0 python src/revamp.py attack.target=stop_sign scenario.sensor_positions.function=use_provided_cam_position attack.iters=100 attack.passes=10 attack.multi_pass_rendering=true attack.samples_per_pixel=64 attack.multi_pass_spp_divisor=8 scene=city`

### Clean-up renders/ predicted renders images

`make clean`

## Pipeline to use an attacked texture, render a batch of images, and predict results. 
Here `TEX_NUM=0` refers to the index of the texture
 
`make TARGET=traffic_light RESULTS_DIR=results/traffic_light TEX_NUM=0 render_predict`

Other params
`TARGET_SCENE = cube
ORIG_TEX = red_tex.png 
TEX_NUM = 0`

e.g., This uses the file `tex_2.png` for the `PERSON` class and render/predict a batch of images.
`make TARGET_TEX=person TEX_NUM=2 render_predict`


### Proccess output logs into loss results - outputs `{filename}.csv` for each unique pass found in the log file.

`python src/results.py -i results/cat/2023-02-10/21-04-51/run.log`

### Process output scores into score results - outputs `{filename}.csv`

`python src/scores.py -i results/person/0_scores.txt -t person`

### Set an (optional alternate) texture before rendering

`make scenes/cube/textures/traffic_light_tex/tex_2.png.set_tex`

or 

`make TARGET_TEX=stop_sign_tex scenes/cube/textures/stop_sign_tex/tex_0.png.set_tex`

### Render batch of images
This command generates renders from 48 sensor positions. See `dt2.py` for details.

Other args to `-cm` 
This generates 264 sensor positions at vertices of 3 concentric half-icospheres

`generate_cube_scene_orbit_cam_positions` 

### Predict on a batch of images

`python src/predict_objdet_batch.py -d red_cube -st 0.3 > results/tf0_scores.txt`

### Unset a texture and use original tex (cube scene)

`make unset_tex`

### Make a movie with `ffmpeg`
`ffmpeg -framerate 30 -i preds/red_cube/render_%d.png -c:v libx264 -preset slow -pix_fmt yuv420p -crf 18 <movie name>.mp4`

## Conventions
Any scene added should at least have the following naming structure:
`<scene name>/<scene name>.xml` e.g., `cube/cube.xml` 

Any texture used should be named as one of `tex_0.png`, `tex_1.png`, etc.
