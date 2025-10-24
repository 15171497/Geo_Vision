
# import os, argparse, torch
# from PIL import Image
# from ..backbones.models_vit import VisionTransformer
# from ..train.finetune_seg import ViTEncoder2D
# from ..heads.segmentation import SimpleSegDecoder
# import torchvision.transforms as T

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--gpus', default=None)
#     ap.add_argument('--img', required=True)
#     ap.add_argument('--ckpt', required=True)
#     ap.add_argument('--img_size', type=int, default=512)
#     ap.add_argument('--num_classes', type=int, default=2)
#     ap.add_argument('--out', default='seg_pred.png')
#     args = ap.parse_args()

#     if args.gpus is not None: os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     vit = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
#     enc = ViTEncoder2D(vit, img_size=args.img_size)
#     dec = SimpleSegDecoder(embed_dim=getattr(vit,'embed_dim',768), out_channels=args.num_classes)
#     sd = torch.load(args.ckpt, map_location='cpu')
#     vit.load_state_dict(sd.get('vit',{}), strict=False); dec.load_state_dict(sd.get('dec',{}), strict=False)
#     vit.to(device); dec.to(device); vit.eval(); dec.eval()

#     t = T.Compose([T.Resize((args.img_size,args.img_size)), T.ToTensor()])
#     x = t(Image.open(args.img).convert('RGB')).unsqueeze(0).to(device)
#     with torch.no_grad():
#         fmap = enc(x)
#         logits = dec(fmap)
#         pred = logits.argmax(1).squeeze(0).cpu().byte()*255
#         Image.fromarray(pred.numpy()).save(args.out)
#         print('Saved', args.out)

# if __name__=='__main__': main()


import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from ..backbones.models_vit import VisionTransformer
from ..train.finetune_seg import ViTEncoder2D  # 直接复用你训练脚本里的健壮编码器
from ..heads.segmentation import SimpleSegDecoder


# ---- 工具：pos_embed 网格插值，保证与训练加载逻辑一致 ----
def _interp_pos_embed_grid(tok_pos: torch.Tensor, new_g: int) -> torch.Tensor:
    """
    tok_pos: [1, N, C] (不含 CLS)，把 N 个 patch 位置从 old_g x old_g 插值到 new_g x new_g。
    返回: [1, new_g*new_g, C]
    """
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
    """
    加载 checkpoint 时，对 sd['pos_embed'] 按新网格大小插值到 new_img_size。
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--img', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--out', default='seg_pred.png')
    ap.add_argument('--restore_original_size', action='store_true',
                    help='将预测结果从 img_size 还原回原图尺寸（最近邻）。')
    args = ap.parse_args()

    # 选择 GPU
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    # 读取并预处理图像
    img = Image.open(args.img).convert('RGB')
    orig_w, orig_h = img.size
    t = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    x = t(img).unsqueeze(0).to(device)  # [1,3,S,S]

    # 推理
    with torch.no_grad():
        fmap = enc(x)             # [1,C,S/16,S/16]
        logits = dec(fmap)        # [1,K,h,w] （可能不是 S×S）

        # 若输出空间不是 S×S，先对齐到 img_size（与训练一致）
        if logits.shape[-2:] != (args.img_size, args.img_size):
            logits = F.interpolate(logits, size=(args.img_size, args.img_size),
                                   mode='bilinear', align_corners=False)

        pred = logits.argmax(1).squeeze(0)  # [S,S]

        # 可选：还原回原图尺寸
        if args.restore_original_size and (orig_h, orig_w) != (args.img_size, args.img_size):
            pred = pred.unsqueeze(0).unsqueeze(0).float()  # [1,1,S,S]
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode='nearest').squeeze(0).squeeze(0)

        # 保存为灰度 PNG（类别 id）
        pred_img = pred.to(torch.uint8).cpu().numpy()
        Image.fromarray(pred_img).save(args.out)
        print('Saved', args.out)


if __name__ == '__main__':
    main()



