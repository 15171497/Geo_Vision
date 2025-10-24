import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

from ..backbones.models_vit import VisionTransformer
from ..train.finetune_seg import ViTEncoder2D
from ..heads.segmentation import SimpleSegDecoder


# -----------------------------
# 工具：pos_embed 网格插值（与训练一致）
# -----------------------------
def _interp_pos_embed_grid(tok_pos: torch.Tensor, new_g: int) -> torch.Tensor:
    assert tok_pos.ndim == 3
    old_n = tok_pos.shape[1]
    C = tok_pos.shape[-1]
    old_g = int(round(old_n ** 0.5))
    if old_g * old_g != old_n or old_g == new_g:
        return tok_pos
    grid = tok_pos.view(1, old_g, old_g, C).permute(0, 3, 1, 2)  # [1,C,H,W]
    grid = F.interpolate(grid, size=(new_g, new_g), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).contiguous().view(1, new_g * new_g, C)  # [1,N',C]
    return grid

def _interpolate_pos_embed_in_state_dict(sd: dict, new_img_size: int, patch: int = 16) -> dict:
    if "pos_embed" not in sd:
        return sd
    pe = sd["pos_embed"]  # [1,1+N,C] 或 [1+N,C]
    if pe.ndim == 2:
        pe = pe.unsqueeze(0)
    cls_pos = pe[:, :1, :]       # [1,1,C]
    tok_pos = pe[:, 1:, :]       # [1,N,C]
    new_g = new_img_size // patch
    tok_pos = _interp_pos_embed_grid(tok_pos, new_g=new_g)
    sd["pos_embed"] = torch.cat([cls_pos, tok_pos], dim=1)  # [1,1+N',C]
    return sd


# -----------------------------
# 调色板：LoveDA 7 类（不含 ignore）
# 0: background, 1: building, 2: road, 3: water, 4: barren, 5: forest, 6: agriculture
# 若 num_classes != 7，会退化为自动生成颜色。
# -----------------------------
def get_palette(num_classes: int, ignore_index: int = -1):
    if num_classes == 7:
        # BGR/HSV 都可，这里给一套易分辨的 RGB
        palette = {
            0: (0, 0, 0),          # background - black
            1: (255, 0, 0),        # building   - red
            2: (128, 64, 128),     # road       - purple-ish
            3: (0, 0, 255),        # water      - blue
            4: (210, 180, 140),    # barren     - tan
            5: (34, 139, 34),      # forest     - forest green
            6: (255, 255, 0),      # agriculture- yellow
        }
    else:
        # 自动生成一些分散的颜色（最多 256）
        palette = {}
        for c in range(num_classes):
            # 生成可区分的颜色：简单哈希
            r = (37 * c) % 255
            g = (17 * c + 80) % 255
            b = (97 * c + 160) % 255
            palette[c] = (r, g, b)
    # ignore 的颜色（可自定义）
    ignore_color = (255, 255, 255)
    return palette, ignore_color

def colorize_mask(mask_np: np.ndarray, palette: dict, ignore_index: int = -1, ignore_color=(255,255,255)) -> Image.Image:
    """mask_np: [H,W] uint8/int，按 palette 映射成彩色 PIL.Image"""
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in palette.items():
        color[mask_np == cls] = rgb
    if ignore_index >= 0:
        color[mask_np == ignore_index] = ignore_color
    return Image.fromarray(color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--list', default='/mnt/public/lyb/Dataset/Semantic_Segmentation/LOVEDA/val_list.txt',
                    help='每行：<image_path> <mask_path>')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--num_classes', type=int, default=7)
    ap.add_argument('--ignore_index', type=int, default=-1,  # LoveDA 的无效像素常用 0，如需忽略可设为 0
                    help='可设为 255 或 0 等；<0 表示不忽略。仅用于彩色可视化时标记忽略色。')
    ap.add_argument('--out_dir', default='seg_val_pred',
                    help='输出目录：将保存 pred_gray / pred_color / label_color 三类结果')
    ap.add_argument('--restore_original_size', action='store_true',
                    help='若开启，将预测结果按标签的原始分辨率保存；否则按 img_size 保存。')
    args = ap.parse_args()

    # 选择 GPU
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 准备输出目录
    out_gray_dir = os.path.join(args.out_dir, 'pred_gray')
    out_color_dir = os.path.join(args.out_dir, 'pred_color')
    out_gt_color_dir = os.path.join(args.out_dir, 'label_color')
    os.makedirs(out_gray_dir, exist_ok=True)
    os.makedirs(out_color_dir, exist_ok=True)
    os.makedirs(out_gt_color_dir, exist_ok=True)

    # 调色板
    palette, ignore_color = get_palette(args.num_classes, ignore_index=args.ignore_index)

    # 模型
    vit = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    enc = ViTEncoder2D(vit, img_size=args.img_size)
    dec = SimpleSegDecoder(embed_dim=getattr(vit, 'embed_dim', 768), out_channels=args.num_classes)

    # 加载权重（含 pos_embed 插值）
    sd = torch.load(args.ckpt, map_location='cpu')
    vit_sd = sd.get('vit', {})
    dec_sd = sd.get('dec', {})

    if isinstance(vit_sd, dict) and len(vit_sd) > 0:
        vit_sd = _interpolate_pos_embed_in_state_dict(vit_sd, new_img_size=args.img_size, patch=16)
        vit.load_state_dict(vit_sd, strict=False)
    if isinstance(dec_sd, dict) and len(dec_sd) > 0:
        dec.load_state_dict(dec_sd, strict=False)

    vit.to(device).eval()
    dec.to(device).eval()

    # 变换
    t = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])

    # 读取 list.txt
    with open(args.list, 'r', encoding='utf-8') as f:
        pairs = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 允许中间有多个空格
            parts = line.split()
            if len(parts) < 2:
                continue
            img_path, mask_path = parts[0], parts[1]
            pairs.append((img_path, mask_path))

    # 遍历推理
    for idx, (img_path, mask_path) in enumerate(pairs):
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARN] 跳过无法读取的图片: {img_path} ({e})")
            continue

        # 读标签（仅为获取尺寸与可视化用）
        gt_np = None
        gt_h, gt_w = None, None
        if os.path.exists(mask_path):
            try:
                gt = Image.open(mask_path)
                gt_np = np.array(gt)  # 直接拿像素值（0..K-1；可能有 ignore）
                gt_h, gt_w = gt_np.shape[:2]
            except Exception as e:
                print(f"[WARN] 标签读取失败，将忽略 label 可视化: {mask_path} ({e})")

        # 预处理 & 推理
        x = t(img).unsqueeze(0).to(device)  # [1,3,S,S]
        with torch.no_grad():
            fmap = enc(x)             # [1,C,S/16,S/16]
            logits = dec(fmap)        # [1,K,h,w]

            # 对齐到目标尺寸：优先用标签大小；否则用 img_size
            if args.restore_original_size and gt_h is not None and gt_w is not None:
                target_size = (gt_h, gt_w)
            else:
                target_size = (args.img_size, args.img_size)

            if logits.shape[-2:] != target_size:
                logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

            pred = logits.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()  # [H,W] uint8

        # 文件名
        base = os.path.splitext(os.path.basename(img_path))[0]

        # 保存预测灰度
        gray_path = os.path.join(out_gray_dir, f"{base}.png")
        Image.fromarray(pred).save(gray_path)

        # 预测彩色
        color_pred = colorize_mask(pred, palette, ignore_index=args.ignore_index, ignore_color=ignore_color)
        color_pred_path = os.path.join(out_color_dir, f"{base}_color.png")
        color_pred.save(color_pred_path)

        # 标签彩色（如果有标签）
        if gt_np is not None:
            # 若目标输出尺寸不是标签原始大小，按 restore_original_size 逻辑是否需要重采样标签可视化
            if not args.restore_original_size and (gt_h is not None and gt_w is not None):
                # 将标签按 img_size 最近邻缩放，仅用于可视化对齐（不改变原文件）
                gt_vis = Image.fromarray(gt_np.astype(np.uint8), mode='L')
                gt_vis = gt_vis.resize((args.img_size, args.img_size), resample=Image.NEAREST)
                gt_np_vis = np.array(gt_vis)
            else:
                gt_np_vis = gt_np

            color_gt = colorize_mask(gt_np_vis, palette, ignore_index=args.ignore_index, ignore_color=ignore_color)
            color_gt_path = os.path.join(out_gt_color_dir, f"{base}_gt_color.png")
            color_gt.save(color_gt_path)

        print(f"[{idx+1}/{len(pairs)}] Saved -> {gray_path}, {color_pred_path}"
              + (f", {color_gt_path}" if gt_np is not None else ""))

    print("全部完成。输出目录：", args.out_dir)


if __name__ == '__main__':
    main()
