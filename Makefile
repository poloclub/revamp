RENDERS = renders
PREDS = preds
PREDS_PREP_DIR = $(RENDERS)/$(PRED_DIR)
RESULTS = results
SCENES = scenes
TARGET_SCENE = cube_scene
TARGET = truck
PRED_DIR = $(TARGET)
TARGET_TEX  = $(TARGET)_tex
ORIG_TEX = noise_tex.png
TEX_NUM = 0
TXT_PREFIX = $(TARGET) # goes ahead of a output text file e.g., "person_scores.txt"
SCENARIOS = scenario_configs
JQ = jq --indent 4 -r
RESULTS_DIR = $(RESULTS)/$(TARGET)
SENSOR_POS_FN = generate_orbit_cam_positions
SCORE_TEST_THRESH=0.3
# generate_32_orbit_cam_positions

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

render_predict: clean
> $(MAKE) $(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/tex_$(TEX_NUM).png.set_tex
> python src/render_batch.py -s $(SCENES)/$(TARGET_SCENE)/$(TARGET_SCENE).xml -cm $(SENSOR_POS_FN) -od $(RENDERS)/$(TARGET)
> python src/predict_objdet_batch.py -d $(TARGET) -st $(SCORE_TEST_THRESH) > $(RESULTS_DIR)/$(TEX_NUM)_scores.txt
> $(MAKE) TARGET_SCENE=$(TARGET_SCENE) unset_tex
> $(MAKE) clean_renders

.PHONY: clean
clean: clean_renders clean_preds

.PHONY: clean_renders
clean_renders:
> rm -f $(RENDERS)/$(TARGET)/*.png

.PHONY: clean_preds
clean_preds:
>  rm -f $(PREDS)/$(TARGET)/*.png

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
