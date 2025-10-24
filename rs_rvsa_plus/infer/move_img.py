#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from pathlib import Path

def safe_copy(src: Path, dst_dir: Path):
    """把 src 复制到 dst_dir 下，若重名则自动添加 _dupN 后缀。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = src.name
    dst = dst_dir / base
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = src.stem
    suf = src.suffix  # includes dot
    n = 1
    while True:
        cand = dst_dir / f"{stem}_dup{n}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
        n += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--list",
        default="/mnt/public/lyb/Dataset/Semantic_Segmentation/LOVEDA/val_list.txt",
        help="验证集列表文件：每行 'image_path mask_path'"
    )
    ap.add_argument(
        "--out_dir",
        default="seg_val_pred",
        help="与你的推理输出同一个根目录；本脚本会在里面新建 originals/ 用来存放原图"
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要复制的文件，不实际复制"
    )
    args = ap.parse_args()

    list_path = Path(args.list)
    out_dir = Path(args.out_dir)
    originals_dir = out_dir / "originals"

    if not list_path.is_file():
        raise FileNotFoundError(f"未找到 list 文件: {list_path}")

    originals_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_path = Path(parts[0])
            if not img_path.is_file():
                print(f"[WARN] 跳过：找不到原图 {img_path}")
                continue
            if args.dry_run:
                print(f"[DRY] copy {img_path} -> {originals_dir}")
            else:
                dst = safe_copy(img_path, originals_dir)
                print(f"[OK] {img_path}  ->  {dst}")
            count += 1

    print(f"\n完成：共处理 {count} 张原图。输出目录：{originals_dir}")

if __name__ == "__main__":
    main()
