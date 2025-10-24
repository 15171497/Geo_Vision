#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
from pathlib import Path

# 配置路径（按你的描述）
IMAGES_ROOT = Path("/mnt/public/lyb/Dataset/Semantic_Segmentation/LOVEDA/JPEGImages")
MASKS_ROOT  = Path("/mnt/public/lyb/Dataset/Semantic_Segmentation/LOVEDA/SegmentationClassAug")
OUT_DIR     = Path("/mnt/public/lyb/Dataset/Semantic_Segmentation/LOVEDA")

# 随机种子与划分比例
SEED = 42
TRAIN_RATIO = 0.8

# 支持的图像扩展名（大小写不敏感）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def main():
    if not IMAGES_ROOT.is_dir():
        print(f"[Error] 图像目录不存在: {IMAGES_ROOT}", file=sys.stderr)
        sys.exit(1)
    if not MASKS_ROOT.is_dir():
        print(f"[Error] 标签目录不存在: {MASKS_ROOT}", file=sys.stderr)
        sys.exit(1)

    # 递归收集所有图像文件
    img_files = [p for p in IMAGES_ROOT.rglob("*") if p.is_file() and is_image_file(p)]
    if not img_files:
        print(f"[Error] 在 {IMAGES_ROOT} 下没有找到图像文件（支持扩展名：{sorted(IMG_EXTS)}）", file=sys.stderr)
        sys.exit(1)

    # 根据相对路径配对标签（默认把后缀改为 .png）
    pairs = []
    missing = []
    for ip in img_files:
        rel = ip.relative_to(IMAGES_ROOT)          # 相对 JPEGImages 的路径
        rel_stem = rel.with_suffix("")             # 去掉扩展名
        mp = MASKS_ROOT / (str(rel_stem) + ".png") # 标签使用 .png

        if mp.is_file():
            pairs.append((str(ip), str(mp)))
        else:
            # 兜底：如果不是 .png，尝试同后缀（极少数数据会这样）
            alt = MASKS_ROOT / rel
            if alt.is_file():
                pairs.append((str(ip), str(alt)))
            else:
                missing.append((str(ip), str(mp)))

    if not pairs:
        print("[Error] 没有配对到任何 image-mask 对。请检查 masks 是否与 images 的相对路径对应，且掩码为 .png。", file=sys.stderr)
        if missing[:10]:
            print("例如（前 10 个）：")
            for a,b in missing[:10]:
                print(" image:", a)
                print(" expect mask:", b)
        sys.exit(1)

    # 打乱并划分
    random.seed(SEED)
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(round(n_total * TRAIN_RATIO))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # 输出文件
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_txt = OUT_DIR / "train_list.txt"
    val_txt   = OUT_DIR / "val_list.txt"

    with train_txt.open("w", encoding="utf-8") as f:
        for ip, mp in train_pairs:
            f.write(f"{ip} {mp}\n")

    with val_txt.open("w", encoding="utf-8") as f:
        for ip, mp in val_pairs:
            f.write(f"{ip} {mp}\n")

    # 打印统计信息
    print("配对完成！")
    print(f"  总样本数: {n_total}")
    print(f"  训练集:   {len(train_pairs)}  -> {train_txt}")
    print(f"  验证集:   {len(val_pairs)}    -> {val_txt}")

    if missing:
        print(f"\n[提示] 有 {len(missing)} 张图像未找到对应的标签（按 .png 推断）。如果确实存在，请检查相对路径或后缀。")
        print("  示例（前 5 条）：")
        for a, b in missing[:5]:
            print("   image:", a)
            print("   expect mask:", b)

if __name__ == "__main__":
    main()
