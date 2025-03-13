#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Description: 将BDD100k数据集转换为COCO格式. COCO数据集的格式如下:
# /home/zyb/YOLOX/datasets/coco128/annotations/instances_val2017.json
#
# @Author: Reuben
# @Date:   2025-03-13
#
# After:
# 1. save the image directory name as: train2017, val2017, test2017.
# 2. save the annotation file name as: instances_train.json, instances_val.json, instances_test.json.
# How to train YOLOX on BDD100k:
# python tools/train.py -n yolox_s_bdd -d 1 -b 8 --fp16 -o [--cache]

import json
import os
import argparse


DEBUG = True

# DATASET_SIZE = "sample" # sample, small, full

DATASET_SCALES = {
    # size: [<image_num_4_train>, <image_num_4_val>, <image_num_4_test>]
    "sample": [70, 10, 20],
    "small": [7000, 1000, 2000],
}


def _gather_categories(
    bdd100k_path, subdir_name, categories, seen_cats, exclude_supercats
):
    sub_dir = os.path.join(bdd100k_path, subdir_name)
    for file in os.listdir(sub_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(sub_dir, file), "r") as f:
            data = json.load(f)
            # Extract categories from objects in frames
            for frame in data["frames"]:
                for obj in frame["objects"]:
                    cat_name = obj["category"]
                    supercat = cat_name.split("/")[0]
                    if supercat in exclude_supercats:
                        continue
                    if cat_name not in seen_cats:
                        seen_cats.add(cat_name)
                        categories.append(
                            {
                                "id": len(seen_cats),  # Auto-generate IDs
                                "name": cat_name,
                                "supercategory": supercat,
                            }
                        )

    return categories, seen_cats


def gather_categories(bdd100k_path):
    categories = []
    seen_cats = set()
    exclude_supercats = set(["area", "lane"])
    categories, seen_cats = _gather_categories(
        bdd100k_path, "val", categories, seen_cats, exclude_supercats
    )
    categories, seen_cats = _gather_categories(
        bdd100k_path, "train", categories, seen_cats, exclude_supercats
    )
    categories, seen_cats = _gather_categories(
        bdd100k_path, "test", categories, seen_cats, exclude_supercats
    )
    return categories


def get_cls_names_in_order(categories):
    # Order the categories by id
    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
    cat_ids = sorted(cat_id_to_name.keys())
    cls_names = [cat_id_to_name[cat_id] for cat_id in cat_ids]
    return cls_names


def cocofmt_annotation_visualize(bdd100k_path, dataset_type="val", save_result=True):
    """Visualize COCO format annotations with bounding boxes"""
    from yolox.utils import vis
    import cv2

    # Load COCO annotations
    anno_file = os.path.join(
        bdd100k_path, "annotations", f"instances_{dataset_type}.json"
    )
    with open(anno_file, "r") as f:
        coco_data = json.load(f)

    # Create output directory
    if save_result:
        output_dir = os.path.join("YOLOX_outputs", "visualize", dataset_type)
        os.makedirs(output_dir, exist_ok=True)

    cls_conf = 0.35
    cls_names = get_cls_names_in_order(coco_data["categories"])
    # Process images
    visual_count = 0
    for img_info in coco_data["images"]:
        if DEBUG and visual_count >= 10:
            break

        # Load image
        img_path = os.path.join(bdd100k_path, dataset_type, img_info["file_name"])
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping...")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}, skipping...")
            continue

        # Get annotations for this image
        annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
        ]

        bboxes = []
        scores = []
        cls = []

        # Draw annotations
        for ann in annotations:
            # Convert COCO bbox to (x1, y1, x2, y2)
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            xyxy = [x1, y1, x1 + w, y1 + h]
            bboxes.append(xyxy)
            scores.append(1.0)
            cls.append(ann["category_id"] - 1)  # COCO format category id is 1-based

        vis_res = vis(img, bboxes, scores, cls, cls_conf, cls_names)

        # Save or show results
        result = vis_res
        if save_result:
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, result)
            print(f"Saved visualization to {output_path}")
        else:
            cv2.imshow("Annotation Preview", result)
            cv2.waitKey(0)

        visual_count += 1

    if save_result:
        print(f"Saved {visual_count} visualizations to {output_dir}")


def convert_bdd100k_dataset_to_coco(
    dataset_path, categories, dataset_type="val", dataset_size=0
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found")

    cocofmt_data = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Converted from BDD100K",
            "date_created": "2025-03-13",
        },
        "licenses": [{"id": 1, "name": "BDD100K License", "url": ""}],
        "categories": categories,
        "type": "instances",
        "images": [],
        "annotations": [],
    }

    # Create category name to ID mapping
    cat_mapping = {cat["name"]: cat["id"] for cat in categories}
    ann_id = 1

    sub_dir = os.path.join(dataset_path, dataset_type)
    for file in os.listdir(sub_dir):
        if not file.endswith(".json"):
            continue

        bdd_json_file = os.path.join(sub_dir, file)
        base_name = os.path.splitext(file)[0]
        # img_file = os.path.join(dataset_type, f"{base_name}.jpg")  # Relative path

        # Add image entry
        img_id = len(cocofmt_data["images"]) + 1
        if dataset_size > 0 and img_id > dataset_size:
            break
        cocofmt_data["images"].append(
            {
                "id": img_id,
                "file_name": f"{base_name}.jpg",
                "height": 720,  # BDD100K fixed size
                "width": 1280,
                "date_captured": "2025-03-13",  # convert date
            }
        )

        with open(bdd_json_file, "r") as f:
            data = json.load(f)
            for frame in data["frames"]:
                for obj in frame["objects"]:
                    # Skip non-box2d and excluded categories
                    if "box2d" not in obj:
                        continue
                    supercat = obj["category"].split("/")[0]
                    if supercat in ["area", "lane"]:
                        continue

                    # Convert bbox format
                    bbox = obj["box2d"]
                    coco_bbox = [
                        bbox["x1"],
                        bbox["y1"],
                        bbox["x2"] - bbox["x1"],  # width
                        bbox["y2"] - bbox["y1"],  # height
                    ]

                    # Add annotation
                    cocofmt_data["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_mapping[obj["category"]],
                            "bbox": coco_bbox,
                            "area": coco_bbox[2] * coco_bbox[3],
                            "iscrowd": 0,
                            "segmentation": [],
                            "attributes": obj.get("attributes", {}),
                        }
                    )
                    ann_id += 1

    # Write output
    anno_file = os.path.join(
        dataset_path, "annotations", f"instances_{dataset_type}.json"
    )
    with open(anno_file, "w") as f:
        json.dump(cocofmt_data, f, indent=2)

    print(f"Converted {ann_id - 1} annotations of {dataset_size} images for {dataset_type} set")


def convert_bdd100k_to_coco(bdd100k_path, dataset_size="full"):
    if not os.path.exists(bdd100k_path):
        raise FileNotFoundError(f"BDD100k path {bdd100k_path} not found")
    if not os.path.exists(os.path.join(bdd100k_path, "train")):
        raise FileNotFoundError(
            f"Train path {os.path.join(bdd100k_path, 'train')} not found"
        )
    if not os.path.exists(os.path.join(bdd100k_path, "val")):
        raise FileNotFoundError(
            f"Val path {os.path.join(bdd100k_path, 'val')} not found"
        )
    if not os.path.exists(os.path.join(bdd100k_path, "test")):
        raise FileNotFoundError(
            f"Test path {os.path.join(bdd100k_path, 'test')} not found"
        )

    # 创建annotations文件夹
    annotations_path = os.path.join(bdd100k_path, "annotations")
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    anno_val_file = os.path.join(annotations_path, "instances_val.json")
    anno_train_file = os.path.join(annotations_path, "instances_train.json")
    anno_test_file = os.path.join(annotations_path, "instances_test.json")

    cats = gather_categories(bdd100k_path)

    if dataset_size == "sample" or dataset_size == "small":
        train_dataset_size = DATASET_SCALES[dataset_size][0]
        val_dataset_size = DATASET_SCALES[dataset_size][1]
        test_dataset_size = DATASET_SCALES[dataset_size][2]
    else:
        train_dataset_size = 0
        val_dataset_size = 0
        test_dataset_size = 0

    convert_bdd100k_dataset_to_coco(bdd100k_path, cats, "val", val_dataset_size)
    convert_bdd100k_dataset_to_coco(bdd100k_path, cats, "train", train_dataset_size)
    convert_bdd100k_dataset_to_coco(bdd100k_path, cats, "test", test_dataset_size)

    print(f"Converted {dataset_size} size of annotations")


def get_args():
    parser = argparse.ArgumentParser(description="Convert BDD100K to COCO format")
    parser.add_argument(
        "--size",
        type=str,
        default="full",
        help="dataset size: sample, small, full",
    )
    parser.add_argument(
        "--visualize", type=bool, default=False, help="visualize annotations"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert_bdd100k_to_coco(
        "/workspace/YOLOX/datasets/bdd100k/", dataset_size=args.size
    )
    if args.visualize:
        cocofmt_annotation_visualize(
            "/workspace/YOLOX/datasets/bdd100k/", "val", save_result=True
        )
