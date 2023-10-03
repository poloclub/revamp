
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

`generate_orbit_cam_positions` 

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

### Troubleshooting

#### Resuming Experiments and Handling Out-of-Memory
Sometimes Mitsuba crashes or you want to add additional passes to an perturbed texture.  To resume / add on to an experiment, follow these steps:
Continue an experiment by adding extra passes with explicit pass names.
1. Set the texture:
` make TARGET=truck TARGET_SCENE=cube scenes/cube/textures/truck_tex/tex_6.png.set_tex`
2. run: `python revamp.py attack_class=truck attack.passes=1 attack.passes_names=[7]`


Rendering large scenes that contain a high number of meshes, it is possible to exhaust the memory of the GPU, resulting in out-of-memory errors.  To solve this problem, you can choose to render at a lower number of samples per pixel (SPP) or lower the resolution of the sensor.  Alternatively, you can render a sequence of images at a lower SPP value while varying the seed of each render and then average the resulting images to compose a lower noise image.  The following command uses `attack.multi_pass_rendering=true`, `attack.samples_per_pixel=64`, and `attack.multi_pass_spp_divisor=8` to use render `64/8=8` images and then averages them together.

`CUDA_VISIBLE_DEVICES=0 python revamp.py attack_class=stop_sign scenario.sensor_positions.function=use_provided_cam_position attack.iters=100 attack.passes=10 attack.multi_pass_rendering=true attack.samples_per_pixel=64 attack.multi_pass_spp_divisor=8 scene=city`