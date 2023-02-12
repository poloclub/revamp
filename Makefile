

RENDERS = renders
PREDS = preds
PRED_DIR = red_cube
PREDS_PREP_DIR = $(RENDERS)/$(PRED_DIR)
RESULTS = results
SCENES = scenes
TARGET_SCENE = cube_scene
TARGET = truck
TARGET_TEX  = $(TARGET)_tex
ORIG_TEX = red_tex.png
TEX_NUM = 0
TXT_PREFIX = $(TARGET) # goes ahead of a output text file e.g., "person_scores.txt"
SCENARIOS = scenario_configs
JQ = jq --indent 4 -r
RESULTS_DIR = $(RESULTS)/$(TARGET)
SENSOR_POS_FN = generate_cube_scene_orbit_cam_positions
# generate_cube_scene_32_orbit_cam_positions

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

render_predict: clean
> $(MAKE) $(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/tex_$(TEX_NUM).png.set_tex
> python src/render_batch.py -s scenes/cube_scene/cube_scene.xml -cm $(SENSOR_POS_FN)
> $(MAKE) img_to_pred
> python src/predict_objdet_batch.py -d red_cube -st 0.3 > $(RESULTS_DIR)/$(TEX_NUM)_scores.txt
> $(MAKE) unset_tex


.PHONY: img_to_pred
img_to_pred: clean_renders_prep_dir mv_img clean_renders

.PHONY: clean
clean: clean_renders clean_preds clean_tex_maps clean_renders_prep_dir clean_preds_prep_dir

.PHONY: clean_renders
clean_renders:
> rm -f $(RENDERS)/*.png

.PHONY: clean_preds
clean_preds:
> rm -f $(PREDS)/*.png

.PHONY: clean_tex_maps
clean_tex_maps:
> rm -f perturbed_tex_map_b*.png

.PHONY: clean_renders_prep_dir
clean_renders_prep_dir:
> rm -f $(PREDS_PREP_DIR)/*.png

.PHONY: clean_preds_prep_dir
clean_preds_prep_dir:
>  rm -f $(PREDS)/$(PRED_DIR)/*.png

.PHONY: mv_img
mv_img:
> cp $(RENDERS)/*.png $(PREDS_PREP_DIR)/

.PHONY: attack_dt2
attack_dt2:
> python src/dt2.py

# This target copies specified texture so it will be used on the target object during rendering
# e.g., `make scenes/cube_scene_c/textures/traffic_light_tex/tex_2.png.set_tex`
# modify directories vars as needed
# this was written to quickly set a texture in a scene

$(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/%.png.set_tex:
> rm -f $(SCENES)/$(TARGET_SCENE)/textures/$(ORIG_TEX)
> cp $(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/$*.png $(SCENES)/$(TARGET_SCENE)/textures/
> mv $(SCENES)/$(TARGET_SCENE)/textures/$*.png $(SCENES)/$(TARGET_SCENE)/textures/$(ORIG_TEX)

.PHONY: unset_tex
unset_tex: 
> rm -f $(SCENES)/$(TARGET_SCENE)/textures/$(ORIG_TEX)
> cp $(SCENES)/$(TARGET_SCENE)/textures/orig_tex/$(ORIG_TEX) $(SCENES)/$(TARGET_SCENE)/textures/$(ORIG_TEX)

.PHONY: stop_sign_results
stop_sign_results:
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=0 render_predict
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=1 render_predict
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=2 render_predict
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=3 render_predict
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=4 render_predict
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=5 render_predict
> $(MAKE) TARGET=stop_sign RESULTS_DIR=results/stop_sign/non_rand_sensor_pos TEX_NUM=6 render_predict
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/0_scores.txt -t stop\ sign
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/1_scores.txt -t stop\ sign
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/2_scores.txt -t stop\ sign
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/3_scores.txt -t stop\ sign
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/4_scores.txt -t stop\ sign
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/5_scores.txt -t stop\ sign
> python src/scores.py -i results/stop_sign/non_rand_sensor_pos/6_scores.txt -t stop\ sign

.PHONY: person_results
person_results:
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=0 render_predict
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=1 render_predict
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=2 render_predict
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=3 render_predict
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=4 render_predict
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=5 render_predict
> $(MAKE) TARGET=person RESULTS_DIR=results/person TEX_NUM=6 render_predict
> python src/scores.py -i results/person/0_scores.txt -t person
> python src/scores.py -i results/person/1_scores.txt -t person
> python src/scores.py -i results/person/2_scores.txt -t person
> python src/scores.py -i results/person/3_scores.txt -t person
> python src/scores.py -i results/person/4_scores.txt -t person
> python src/scores.py -i results/person/5_scores.txt -t person
> python src/scores.py -i results/person/6_scores.txt -t person




.PHONY: bicycle_results
bicycle_results:
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/0_scores.txt -t bicycle
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/1_scores.txt -t bicycle
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/2_scores.txt -t bicycle
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/3_scores.txt -t bicycle
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/4_scores.txt -t bicycle
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/5_scores.txt -t bicycle
> python src/scores.py -i results/bicycle/rand_sensor_pos/2023-02-11/18-52-39/6_scores.txt -t bicycle


.PHONY: traffic_light_results
traffic_light_results:
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/0_scores.txt -t traffic\ light
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/1_scores.txt -t traffic\ light
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/2_scores.txt -t traffic\ light
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/3_scores.txt -t traffic\ light
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/4_scores.txt -t traffic\ light
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/5_scores.txt -t traffic\ light
> python src/scores.py -i results/traffic_light/rand_sensor_pos/2023-02-11/17-51-05/6_scores.txt -t traffic\ light


.PHONY: car_results
car_results:
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/0_scores.txt -t car
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/1_scores.txt -t car
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/2_scores.txt -t car
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/3_scores.txt -t car
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/4_scores.txt -t car
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/5_scores.txt -t car
> python src/scores.py -i results/car/rand_sensor_pos/2023-02-11/16-45-49/6_scores.txt -t car
