# MixPHM

## Overview

Pytorch implementation of MixPHM described in the manuscript (This implementation is based on [VL-T5](https://github.com/j-min/VL-T5)). 

## Installation

```shell
pip install -r requirements.txt
```

## Datasets

Please see [data/README.md](data/README.md) to prepare datasets.

```angular2html
├── data
│   ├── annotation
│   │   ├── answer_list.json
│   │   ├── gqa
│   │   │   ├── testdev.json
│   │   │   ├── train.json
│   │   │   ├── trainval_ans2label.json
│   │   │   ├── trainval_label2ans.json
│   │   │   └── valid.json
│   │   ├── lxmert_split
│   │   │   ├── minival.json
│   │   │   ├── nominival.json
│   │   │   ├── test.json
│   │   │   ├── train.json
│   │   │   └── val.json
│   │   ├── okvqa
│   │   │   ├── mscoco_train2014_annotations.json
│   │   │   ├── mscoco_val2014_annotations.json
│   │   │   ├── train.json
│   │   │   ├── trainval_ans2label.json
│   │   │   ├── trainval_label2ans.json
│   │   │   └── val.json
│   │   └── vqav2
│   │       ├── trainval_ans2label.json
│   │       ├── trainval_label2ans.json
│   │       ├── v2_mscoco_train2014_annotations.json
│   │       ├── v2_mscoco_val2014_annotations.json
│   │       └── val.json
│   ├── coco_imgfeat
│   │       ├── train_obj36.h5
│   │       └── val_obj36.h5
│   └── vg_imgfeat
│   │       ├── vg_gqa_obj36.h5
│   │       └── gqa_testdev_obj36.h5
```

## Parameter-efficient tuning and testing

- Experiments on VQA v2 dataset.
```shell
bash scripts/vqav2_vlt5_mixphm.sh
```
- Experiments on GQA dataset.
```shell
bash scripts/gqa_vlt5_mixphm.sh
```
- Experiments on OK-VQA dataset.
```shell
bash scripts/okvqa_vlt5_mixphm.sh
```

<!-- ## Tuned weights -->
