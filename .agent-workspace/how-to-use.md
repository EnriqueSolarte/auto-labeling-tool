
This is a rough and simple description of how to use the current implementation. 

1. Create mask by manual prompting.
```
python3 create_masks_by_manual_prompts.py \
--sequence-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/cam_0/ \
--output-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/
```
Then an image will appear. Then the user is prompt to select the initial frame. After select the initial frame the user will be prompt to define some input prompt point that are within the object in the image. Then the system will use SAM3 to propagate mask sequentially. The mask are stored at --output-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/cam_0/<date-format-data>

```
masks/
`-- cam_0
    |-- 20260514_054016
    `-- 20260514_054309
```

2. Mask Aggregation
The main point of the auto-labeling is to easily collect as many masks as possible from multiple iterations in point (1). To aggregate all of this mask, we use the next command: 
```
python3 aggregate_masks.py \
--masks-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/cam_0/ --debug /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/cam_0/
```
all mask defined for masks/cam_0/ will be aggregate
```
masks/
`-- cam_0
    |-- 20260514_054016
    |-- 20260514_054309
    `-- average_mask
```

3. Create a dataset formal Ultralitcs YOLO 
```
python3 create_training_dataset_from_masks.py \
--masks_dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/masks/cam_0/average_mask/ \
--images_dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/cam_0/ \
--output-root /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/labels/ \
--class-name bamboo
```
The output looks like 
```
labels/
`-- 20260514_055605
    |-- data.yaml
    |-- images
    `-- labels
```

4. Check the dataset stats
```
 python3 dataset_stats.py --dataset-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/labels/20260514_055605/ 
```

5. To reshuffle a dataset
```
python3 manipulate_dataset.py --dataset-dir /data/panda/panda-v3-data/panda-v3-2026-05-12/_5/labels/20260514_055605/ --ratios 0.5 0.25 
```
