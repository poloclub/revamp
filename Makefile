

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

# Taken from https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

render_predict: clean
> $(MAKE) $(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/tex_$(TEX_NUM).png.set_tex
> python src/render_batch.py -s scenes/cube_scene/cube_scene.xml -cm generate_cube_scene_orbit_cam_positions
> $(MAKE) img_to_pred
> python src/predict_objdet_batch.py -d red_cube -st 0.3 > results/$(TARGET)/$(TEX_NUM)_scores.txt
> $(MAKE) $(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/tex_$(TEX_NUM).png.unset_tex


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

# running same target as above except use '.unset' will remove the copy of the tex and replace the original tex
$(SCENES)/$(TARGET_SCENE)/textures/$(TARGET_TEX)/%.png.unset_tex:
> rm -f $(SCENES)/$(TARGET_SCENE)/textures/$(ORIG_TEX)
> cp $(SCENES)/$(TARGET_SCENE)/textures/orig_tex/$(ORIG_TEX) $(SCENES)/$(TARGET_SCENE)/textures/$(ORIG_TEX)


$(SCENARIOS)/%.json.run: $(SCENARIOS)/%.json
> python src/run.py $(*D)/$<

# use COCO 2017 train split for DT2 attacks
$(SCENARIOS)/coco.json: $(SCENARIOS)/baseline.json
> cat $< | $(JQ) '.dataset.name = "coco_2017_train"' > $@

# specify any COCO label as a string e.g., "sports_ball" or "person"
$(SCENARIOS)/coco_%.json: $(SCENARIOS)/coco.json
>	cat $< | $(JQ) '.attack.kwargs.target = "$*"'  \
         | $(JQ) '.sysconfig.output_path = "results/$*"' > $@