#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Description: 将BDD100k数据集转换为COCO格式. COCO数据集的格式如下:
# /home/zyb/YOLOX/datasets/coco128/annotations/instances_val2017.json
# 
# @Author: Reuben
# @Date:   2025-03-13 
# 
import json
import os


def _gather_categories(bdd100k_path, subdir_name, categories, seen_cats, exclude_supercats):
    sub_dir = os.path.join(bdd100k_path, subdir_name)
    for file in os.listdir(sub_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(sub_dir, file), 'r') as f:
            data = json.load(f)
            # Extract categories from objects in frames
            for frame in data["frames"]:
                for obj in frame["objects"]:
                    cat_name = obj["category"]
                    supercat = cat_name.split('/')[0]
                    if supercat in exclude_supercats:
                        continue
                    if cat_name not in seen_cats:
                        seen_cats.add(cat_name)
                        categories.append({
                            "id": len(seen_cats),  # Auto-generate IDs
                            "name": cat_name,
                            "supercategory": supercat
                        })

    return categories, seen_cats


def gather_categories(bdd100k_path):
    categories = []
    seen_cats = set()
    exclude_supercats = set(["area", "lane"])
    categories, seen_cats = _gather_categories(bdd100k_path, "val", categories, seen_cats, exclude_supercats)
    categories, seen_cats = _gather_categories(bdd100k_path, "train", categories, seen_cats, exclude_supercats)
    categories, seen_cats = _gather_categories(bdd100k_path, "test", categories, seen_cats, exclude_supercats)
    return categories


def convert_bdd100k_dataset_to_coco(dataset_path, categories, dataset_type="val"):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} not found")
    
    cocofmt_data = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Converted from BDD100K",
            "date_created": "2025-03-13"
        },
        "licenses": [{
            "id": 1,
            "name": "BDD100K License",
            "url": ""
        }],
        "categories": categories,
        "type": "instances",
        "images": [],
        "annotations": []
    }

    # output coco format annotations file
    anno_file = os.path.join(dataset_path, "annotations", f"instances_{dataset_type}.json")
    with open(anno_file, 'w') as f:
        json.dump(cocofmt_data, f, indent=2)

    print(f"Convert {dataset_type} dataset to coco format annotations file done, saved to {anno_file}")
    

def convert_bdd100k_to_coco(bdd100k_path):
    if not os.path.exists(bdd100k_path):
        raise FileNotFoundError(f"BDD100k path {bdd100k_path} not found")
    if not os.path.exists(os.path.join(bdd100k_path, "train")):
        raise FileNotFoundError(f"Train path {os.path.join(bdd100k_path, 'train')} not found")
    if not os.path.exists(os.path.join(bdd100k_path, "val")):
        raise FileNotFoundError(f"Val path {os.path.join(bdd100k_path, 'val')} not found")
    if not os.path.exists(os.path.join(bdd100k_path, "test")):
        raise FileNotFoundError(f"Test path {os.path.join(bdd100k_path, 'test')} not found")
    
    # 创建annotations文件夹
    annotations_path = os.path.join(bdd100k_path, "annotations")
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    anno_val_file = os.path.join(annotations_path, "instances_val.json")
    anno_train_file = os.path.join(annotations_path, "instances_train.json")
    anno_test_file = os.path.join(annotations_path, "instances_test.json")

    cats = gather_categories(bdd100k_path)

    convert_bdd100k_dataset_to_coco(bdd100k_path, cats, "val")
    pass


if __name__ == "__main__":
    convert_bdd100k_to_coco(
        "/workspace/YOLOX/datasets/bdd100k/"
    )


