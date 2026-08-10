Date: 2026-05-18
Project version: v1.1.0
Description: Dataset tool action items define input, sampling, split, and label changes.

1. One inconsistency worth calling out: the help text says “binary images or .npy stacks,” but the actual file discovery only includes .npy. So unless another caller passes image masks directly into internals, the standalone script will not discover .png mask files despite what its help text claims.
* ACTION: Let's define only .npy for mask definitions. the png mask are for visualization purposes only

2. Random dataset sampling? No.
It uses all matched mask/image pairs. It only randomizes the train/val split with a seed. It does not let you say “sample 500 images” or “sample 20% of masks.” It also does not create test, valid, split-1, split-2, etc. It creates only:
images/train
images/val
labels/train
labels/val
* ACTION: Let's add this feature: We will give the option of sampling a percentage of the raw data.
* ACTION: Le'ts add this feature: We will give a set of split names by default the set is "train", "val", but in practice could be as many as the user required. Together with this, we will give a set of ratios, by default 0.7, 0.3

3. Masks or bbox labels? Only bounding boxes.
The input is masks, but the output labels are YOLO detection boxes only:
class_id center_x center_y width height
It converts each mask slice into one normalized bounding box. It does not write YOLO segmentation polygons, does not preserve masks in the dataset, and does not create mask labels.
* ACTION: Let's add the option to create either bbx labels or mask labels.
* ACTION: Let's add the option of copy or create symb links for the images (default symb links)
