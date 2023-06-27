## Datasets


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

- Download annotation files from [here](https://drive.google.com/file/d/1YgrCi-rty4bkBu73EvDEdepr5PafwL59/view?usp=sharing). 


- Download pre-extracted image features [train_obj36.h5](https://drive.google.com/drive/folders/17rpBqULQKEBUeAmUuNH_ctGXGDGqYZ2_) and [val_obj36.h5](https://drive.google.com/drive/folders/17rpBqULQKEBUeAmUuNH_ctGXGDGqYZ2_) of COCO images, and saving them ```data/coco_imgfeat```. 


- Refer to [here](https://github.com/j-min/VL-T5/tree/main/feature_extraction) to obtain ```vg_gqa_obj36.h5```, directly download pre-extracted image features [gqa_testdev_obj36.h5](https://drive.google.com/file/d/14KbtFIcPFJnl2j-J0raqm8-XDxZTOCTG/view?usp=sharing) of VG images for GQA, and saving them ```data/vg_imgfeat```. 


