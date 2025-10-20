# SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction

*Accepted for publication at NeurIPS 2025*

[Fabian Immel](https://scholar.google.com/citations?hl=de&user=uHlmd9QAAAAJ&view_op=list_works&sortby=pubdate)<sup>1 :email:</sup> , [Jan-Hendrik Pauls](https://scholar.google.com/citations?user=0LbD7HUAAAAJ&hl=de&oi=ao)<sup>2</sup>, [Richard Fehler](https://scholar.google.com/citations?hl=de&user=gOQYH4AAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup> , [Frank Bieder](https://scholar.google.com/citations?user=mAMWuMUAAAAJ&hl=de&oi=ao)<sup>1</sup> , [Jonas Merkert](https://scholar.google.de/citations?user=lv_OG7MAAAAJ&hl=de&oi=ao)<sup>2</sup> , [Christoph Stiller](https://scholar.google.com/citations?user=OeAQ2c0AAAAJ&hl=de&oi=ao)<sup>2</sup>
 
<sup>1</sup> FZI Research Center for Information Technology <sup>2</sup> Institute for Measurement and Control Systems, Karlsruhe Institute of Technology

(<sup>:email:</sup>) corresponding author

### [Project Page :globe_with_meridians:](https://immel-f.github.io/SDTagNet/)

### [ArXiv Preprint](https://arxiv.org/abs/2506.08997)

Official implementation of `SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction`

### Introduction

![overview](figs/architecture_overview.png "overview")

Autonomous vehicles rely on detailed and accurate environmental information to operate safely. High definition (HD) maps offer a promising solution, but their high maintenance cost poses a significant barrier to scalable deployment. This challenge is addressed by online HD map construction methods, which generate local HD maps from live sensor data. However, these methods are inherently limited by the short perception range of onboard sensors. To overcome this limitation and improve general performance, recent approaches have explored the use of standard definition (SD) maps as prior, which are significantly easier to maintain. We propose SDTagNet, the first online HD map construction method that fully utilizes the information of widely available SD maps, like OpenStreetMap, to enhance far range detection accuracy. Our approach introduces two key innovations. First, in contrast to previous work, we incorporate not only polyline SD map data with manually selected classes, but additional semantic information in the form of textual annotations. In this way, we enrich SD vector map tokens with NLP-derived features, eliminating the dependency on predefined specifications or exhaustive class taxonomies. Second, we introduce a point-level SD map encoder together with orthogonal element identifiers to uniformly integrate all types of map elements. Experiments on Argoverse 2 and nuScenes show that this boosts map perception performance by up to +5.9 mAP (+45%) w.r.t. map construction without priors and up to +3.2 mAP (+20%) w.r.t. previous approaches that already use SD map priors.

## Model Checkpoints

You can find trained checkpoints of SDTagNet and its NLP module, the OSM maps as well as the OSM tags used for pretraining [here](https://huggingface.co/datasets/immel-f/SDTagNet/tree/main) on HuggingFace.

## Basic Usage

The environment can be found in the Dockerfile in the folder, simply build the corresponding docker image, which will contain dependencies and a copy of this codebase.

The code follows the structure of the M3TR codebase, so the basic workflow is similar. 
After downloading the Argoverse 2 or nuScenes dataset, you need to create the labels for training:

```
python tools/sdtagnet/custom_av2_map_converter.py --data-root /datasets/public/argoverse20/sensor --osm-map-root av2_osm_maps_all --out-root ./gen_labels/av2_no_prior_with_osm_map --masked-
elements divider_dashed divider_solid boundary centerline ped_crossing --nproc 96 --pc-range -60.0 -30.0 -5.0 30.0 15.0 3.0
```

*(nuScenes command similar)*

The command above would generate the labels for the far range setting (`--pc-range` parameters). The `--masked-elements` parameters are from M3TR and in this case just mean that we do not use any HD map prior. For the MapTracker GT setting on Argoverse 2, you can use the following script with the same syntax:

```
python tools/sdtagnet/custom_av2_map_converter_maptracker_gt.py --data-root /datasets/public/argoverse20/sensor --osm-map-root av2_osm_maps_all --out-root ./gen_labels/av2_maptracker_gt_wit
h_osm_map_100_50 --masked-elements divider boundary ped_crossing --nproc 64 --pc-range -50.0 -25.0 -5.0 50.0 25.0 3.0
```



**Note:** The labels generated in this way follow the default split from MapTRv2. To mirror the evaluation in the paper and use the geographic split, follow the instructions in [geographical-splits](https://github.com/LiljaAdam/geographical-splits) for MapTRv2 using our forked version of `geographical-splits` in this repository. 
This forked version also includes a `convert_pkls_streammapnet.py` script to convert pickle files to the StreamMapNet geo split.

To train a model, use the `dist_train.sh` script: 

```
./tools/dist_train.sh ./projects/configs/sdtagnet/sdtagnet_av2_3d_r50_24ep_120_60_range.py 4
```

This command would train the far range model for Argoverse 2 on 4 GPUs. Note that you need to need to have a valid `nlp_model_path` and `ann_root` in the config.

## Acknowledgements

This project is based on the codebase of M3TR:

* [M3TR](https://github.com/immel-f/m3tr) 

We're also grateful for the open-source codebase of MapTRv2, which formed the basis of M3TR:

* [MapTRv2](https://github.com/hustvl/MapTR/tree/maptrv2) 
