# REVAMP: Automated Simulations of Adversarial Attacks on Arbitrary Objects in Realistic Scenes
[![MIT license](http://img.shields.io/badge/license-BSD--3-brightgreen.svg)](http://opensource.org/licenses/MIT)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2110.11227-b3131b.svg)](https://arxiv.org/abs/xxxx.xxxxx) -->

https://github.com/poloclub/revamp/assets/683979/97fb09e2-f672-42fb-ba61-e23f57975ab0

Deep Learning models, such as those used in an autonomous vehicle are vulnerable to adversarial attacks where an attacker could place an adversarial object in the environment, leading to mis-classification. Generating these adversarial objects in the digital space has been extensively studied, however successfully transferring these attacks from the digital real to the physical real has proven challenging when controlling for real-world environmental factors. In response to these limitations, we introduce REVAMP, an easy to use python library that is the first-of-its-kind tool for creating attack scenarios with arbitrary objects and simulating realistic lighting and environmental factors, lighting, reflection, and refraction. REVAMP enables researchers and practitioners to swiftly explore various scenarios within the digital realm by offering a wide range of configurable options for designing experiments and using differentiable rendering to reproduce physically plausible adversarial objects.

## REVAMP is easy to use!  
`python revamp.py scene=city texture=mail_box attack_class=stop_sign multicam=64`

Running this command chooses the "city" scene from the library of scenes, designates the texture on the mailbox as the attackable parameter, and sets the desired attack class to "stop sign" and uses 64 unique camera positions for rendering.


![crown_jewel](https://github.com/matthewdhull/diff_rendering_attack/assets/683979/95dc6b8e-a948-4989-b3da-951e94ad4c72)

## Getting Started

`conda env create -f environment.yml`

Install Detectron2

`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

### Model Weights
We use [Robust ImageNet Models](https://huggingface.co/madrylab/robust-imagenet-models). You'll need to choose an appropriate model for your experiment. Currently we use this ResNet-50 [L2-Robust Model](https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_l2_eps0.05.ckpt) for object detection with Detectron2. After downloading this model, place it in the `pretrained-models/` directory.  If you want another model, you'll need to create a model config in `configs/model/{model}.yaml`. You may copy the existing configs and use it as a template.

### Examples
Run a texture attack on Detectron2 and log the results to a file.  We use Hydra for configuring experiments and you can easily supply your own Hydra-style config arguments. See this [Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)

#### Specify Target Class and Camera Positioning
`python revamp.py scene=city texture=mail_box attack_class=stop_sign multicam=64`

#### Specify Target Class and Use a Different Scene
`python revamp.py scene=mesa texture=mesa attack_class=bus multicam=1`

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



# Credits
Led by [Matthew Hull](https://matthewdhull.github.io), REVAMP was created in a collaboaration with  [Zijie J. Wang](https://zijie.wang) and [Duen Horng Chau](https://poloclub.github.io/polochau/).

<!-- # Citation
To learn more about REVAMP, please read our preliminary two-page [demo paper](https://arxiv.org/abs/2110.11227). Thanks!

```latex
@inproceedings{hull2021autogradeviz,
      title={Towards Automatic Grading of D3.js Visualizations},
      author={Matthew Hull, Connor Guerin, Justin Chen, Susanta Routray, Duen Horng (Polo) Chau},
      booktitle = {IEEE Visualization Conference (VIS), Poster},
      year={2021}}
``` -->
