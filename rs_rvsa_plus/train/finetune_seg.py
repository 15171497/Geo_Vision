
# import os, argparse, torch, torch.nn as nn
# from torch.utils.data import DataLoader
# from ..utils.common import set_seed, select_gpus, is_main_process
# from ..data.datasets import SegmentationDataset
# from ..backbones.models_vit import VisionTransformer
# from ..heads.segmentation import SimpleSegDecoder

# class ViTEncoder2D(nn.Module):
#     def __init__(self, vit: VisionTransformer, img_size: int, patch: int=16):
#         super().__init__()
#         self.vit = vit
#         self.img_size = img_size; self.patch = patch
#     def forward(self, x):
#         B = x.size(0)
#         feats = self.vit.forward_features(x)
#         C = feats.size(-1)
#         n = (self.img_size//self.patch)**2
#         fmap = feats[:,1:1+n,:].transpose(1,2).reshape(B, C, self.img_size//self.patch, self.img_size//self.patch)
#         return fmap

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--gpus', default=None)
#     ap.add_argument('--train', required=True)
#     ap.add_argument('--val', required=True)
#     ap.add_argument('--num_classes', type=int, default=2)
#     ap.add_argument('--epochs', type=int, default=80)
#     ap.add_argument('--batch_size', type=int, default=4)
#     ap.add_argument('--img_size', type=int, default=512)
#     ap.add_argument('--ignore_index', type=int, default=255,
#                 help='未标注像素的忽略值，常用 255；设为 -1 关闭忽略')
#     ap.add_argument('--lr', type=float, default=3e-4)
#     ap.add_argument('--backbone_ckpt', default=None)
#     ap.add_argument('--train_ratio', type=float, default=0.2)
#     ap.add_argument('--out', default='./outputs_seg')
#     args = ap.parse_args()

#     select_gpus(args.gpus); set_seed(42)
#     os.makedirs(args.out, exist_ok=True)

#     tr = SegmentationDataset(args.train, img_size=args.img_size, num_classes=args.num_classes)
#     va = SegmentationDataset(args.val, img_size=args.img_size, num_classes=args.num_classes)
#     dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=8)
#     dl_va = DataLoader(va, batch_size=1, shuffle=False, num_workers=4)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     vit = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
#     if args.backbone_ckpt and os.path.exists(args.backbone_ckpt):
#         sd = torch.load(args.backbone_ckpt, map_location='cpu')
#         sd = sd.get('model', sd)
#         vit.load_state_dict(sd, strict=False)
#     enc = ViTEncoder2D(vit, img_size=args.img_size)
#     dim = getattr(vit,'embed_dim',768)
#     dec = SimpleSegDecoder(embed_dim=dim, out_channels=args.num_classes)
#     model = nn.Module()
#     model.enc = enc; model.dec = dec
#     model.to(device)

#     params = list(vit.parameters()); n=len(params); k=int(round(n*args.train_ratio))
#     for i,p in enumerate(params):
#         p.requires_grad = i >= (n-k)
#     opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
#     # loss_fn = nn.CrossEntropyLoss()
#     loss_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index)


#     for epoch in range(args.epochs):
#         model.train()
#         for x,y,_ in dl_tr:
#             x,y = x.to(device), y.to(device)
#             fmap = model.enc(x)
#             logits = model.dec(fmap)
#             loss = loss_fn(logits, y)
#             opt.zero_grad(); loss.backward(); opt.step()
#         model.eval(); print(f"Epoch {epoch+1} done.")

#     if is_main_process():
#         os.makedirs(args.out, exist_ok=True)
#         torch.save({'vit':vit.state_dict(),'dec':dec.state_dict()}, os.path.join(args.out,'last_seg.pth'))

# if __name__=='__main__': main()

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.common import set_seed, select_gpus, is_main_process
from ..data.datasets import SegmentationDataset
from ..backbones.models_vit import VisionTransformer
from ..heads.segmentation import SimpleSegDecoder


# ----------------------------
# 工具：pos_embed 网格插值
# ----------------------------
def _interp_pos_embed_grid(tok_pos: torch.Tensor, new_g: int) -> torch.Tensor:
    """
    tok_pos: [1, N, C] (不含 CLS)，把其中 N 个 patch 位置从 old_g x old_g 插值到 new_g x new_g。
    返回: [1, new_g*new_g, C]
    """
    assert tok_pos.ndim == 3
    old_n = tok_pos.shape[1]
    C = tok_pos.shape[-1]
    old_g = int(round(old_n ** 0.5))
    if old_g * old_g != old_n or old_g == new_g:
        return tok_pos  # 不是方阵或无需插值则直接返回
    grid = tok_pos.view(1, old_g, old_g, C).permute(0, 3, 1, 2)  # [1, C, H, W]
    grid = F.interpolate(grid, size=(new_g, new_g), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).contiguous().view(1, new_g * new_g, C)  # [1, N', C]
    return grid


def _interpolate_pos_embed_in_state_dict(sd: dict, new_img_size: int, patch: int = 16) -> dict:
    """
    加载 checkpoint 时，对 sd['pos_embed'] 按新网格大小插值到 new_img_size。
    """
    if "pos_embed" not in sd:
        return sd
    pe = sd["pos_embed"]  # [1, 1+N, C] 或 [1+N, C]
    if pe.ndim == 2:
        pe = pe.unsqueeze(0)
    cls_pos = pe[:, :1, :]          # [1,1,C]
    tok_pos = pe[:, 1:, :]          # [1,N,C]
    new_g = new_img_size // patch
    tok_pos = _interp_pos_embed_grid(tok_pos, new_g=new_g)
    sd["pos_embed"] = torch.cat([cls_pos, tok_pos], dim=1)  # [1, 1+N', C]
    return sd


# ----------------------------
# ViT -> 2D 编码器（鲁棒返回 [B,C,H/16,W/16]）
# ----------------------------
class ViTEncoder2D(nn.Module):
    """
    把 ViT 的 token 转成 2D feature map，健壮处理：无论 forward_features 返回 [B, C] 还是 [B, 1+N, C] 都能工作。
    """
    def __init__(self, vit: VisionTransformer, img_size: int, patch: int = 16):
        super().__init__()
        self.vit = vit
        self.img_size = img_size
        self.patch = patch

    def _forward_tokens_manual(self, x: torch.Tensor) -> torch.Tensor:
        """
        手动从 vit 里拿 patch_embed/cls/pos_embed/blocks/norm，返回 [B, 1+N, C]。
        同时对 pos_embed 做必要的插值以适配当前 grid。
        """
        vit = self.vit
        B = x.size(0)

        # patch embed
        x_tok = vit.patch_embed(x)  # [B, N, C]
        N = x_tok.shape[1]
        new_g = int(round(N ** 0.5))

        # cls token
        if hasattr(vit, "cls_token") and vit.cls_token is not None:
            cls_tok = vit.cls_token.expand(B, -1, -1)  # [B,1,C]
        else:
            cls_tok = torch.zeros(B, 1, x_tok.size(-1), device=x_tok.device, dtype=x_tok.dtype)

        # 位置编码
        if hasattr(vit, "pos_embed") and vit.pos_embed is not None:
            pos = vit.pos_embed  # [1, 1+N0, C] 可能与当前 N 不同
            if pos.ndim == 2:
                pos = pos.unsqueeze(0)
            pos_cls, pos_tok = pos[:, :1, :], pos[:, 1:, :]
            pos_tok = _interp_pos_embed_grid(pos_tok, new_g=new_g)
            pos = torch.cat([pos_cls, pos_tok], dim=1)  # [1, 1+N, C]
        else:
            pos = None

        x_all = torch.cat((cls_tok, x_tok), dim=1)  # [B, 1+N, C]
        if pos is not None and pos.shape[1] == x_all.shape[1]:
            x_all = x_all + pos

        # pos_drop
        if hasattr(vit, "pos_drop") and vit.pos_drop is not None:
            x_all = vit.pos_drop(x_all)

        # blocks
        for blk in vit.blocks:
            x_all = blk(x_all)

        # norm
        if hasattr(vit, "norm") and vit.norm is not None:
            x_all = vit.norm(x_all)

        return x_all  # [B, 1+N, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # 尝试：新版 timm 支持 return_all_tokens=True
        feats = None
        try:
            feats = self.vit.forward_features(x, return_all_tokens=True)
        except TypeError:
            try:
                feats = self.vit.forward(x, return_all_tokens=True)
            except TypeError:
                pass

        if feats is None or feats.ndim == 2:  # 失败或只给了 CLS
            feats = self._forward_tokens_manual(x)  # [B, 1+N, C]

        C = feats.size(-1)
        n = (self.img_size // self.patch) ** 2
        if feats.shape[1] < 1 + n:  # token 不足时，强制走手工路径
            feats = self._forward_tokens_manual(x)

        fmap = (feats[:, 1:1 + n, :]     # [B, N, C]
                .transpose(1, 2)         # [B, C, N]
                .reshape(B, C, self.img_size // self.patch, self.img_size // self.patch))
        return fmap


# ----------------------------
# 简易评估（像素精度 + 总体 IoU 近似）
# ----------------------------
@torch.no_grad()
def _evaluate(model: nn.Module, dl, device, num_classes: int, ignore_index: int):
    model.eval()
    correct = 0
    labeled = 0
    inter = 0
    union = 0

    for x, y, _ in dl:
        x = x.to(device)
        y = y.to(device)  # [B,H,W]

        logits = model.dec(model.enc(x))  # [B,K,h,w]
        # 若与 y 空间不一致，插值到 y 的大小
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

        pred = logits.argmax(1)  # [B,H,W]

        if ignore_index >= 0:
            mask = (y != ignore_index)
            correct += (pred.eq(y) & mask).sum().item()
            labeled += mask.sum().item()
            inter += ((pred == y) & mask).sum().item()
            union += ((pred == y) | mask).sum().item()
        else:
            correct += (pred.eq(y)).sum().item()
            labeled += y.numel()
            inter += (pred == y).sum().item()
            union += y.numel()

    pix_acc = correct / max(labeled, 1)
    miou_like = inter / max(union, 1)
    return pix_acc, miou_like


# ----------------------------
# 训练主函数
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--ignore_index', type=int, default=255,
                    help='未标注像素的忽略值（如 255）。设为 <0 关闭忽略。')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone_ckpt', default=None)
    ap.add_argument('--train_ratio', type=float, default=0.2,
                    help='解冻比例（0~1），仅训练 ViT 参数列表中最后该比例的参数')
    ap.add_argument('--out', default='./outputs_seg')
    args = ap.parse_args()

    # 环境
    select_gpus(args.gpus)
    set_seed(42)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据
    tr = SegmentationDataset(args.train, img_size=args.img_size, num_classes=args.num_classes)
    va = SegmentationDataset(args.val, img_size=args.img_size, num_classes=args.num_classes)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dl_va = DataLoader(va, batch_size=1, shuffle=False, num_workers=4)

    # 模型
    vit = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)

    # 载入预训练 + pos_embed 插值
    if args.backbone_ckpt and os.path.exists(args.backbone_ckpt):
        sd = torch.load(args.backbone_ckpt, map_location='cpu')
        sd = sd.get('model', sd)
        sd = _interpolate_pos_embed_in_state_dict(sd, new_img_size=args.img_size, patch=16)
        vit.load_state_dict(sd, strict=False)

    enc = ViTEncoder2D(vit, img_size=args.img_size, patch=16)
    dim = getattr(vit, 'embed_dim', 768)
    dec = SimpleSegDecoder(embed_dim=dim, out_channels=args.num_classes)

    model = nn.Module()
    model.enc = enc
    model.dec = dec
    model.to(device)

    # 冻结策略：仅训练最后 train_ratio 比例的 ViT 参数 + 全部解码器参数
    params = list(vit.parameters())
    n = len(params)
    k = int(round(n * args.train_ratio))
    for i, p in enumerate(params):
        p.requires_grad = i >= (n - k)

    # 优化 & 损失
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

    best_miou = -1.0
    for epoch in range(args.epochs):
        model.train()
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)  # y: [B,H,W]，整型类别图

            logits = model.dec(model.enc(x))   # [B,K,h,w]
            # 关键修复：对齐到标签分辨率
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)

            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # 评估
        pix_acc, miou_like = _evaluate(model, dl_va, device, args.num_classes, args.ignore_index)
        if is_main_process():
            print(f"Epoch {epoch+1}/{args.epochs} | pixAcc={pix_acc:.4f} | mIoU~={miou_like:.4f}")
            # 保存最好
            if miou_like > best_miou:
                best_miou = miou_like
                torch.save({'vit': vit.state_dict(), 'dec': dec.state_dict()},
                           os.path.join(args.out, 'best_seg.pth'))

    # 保存最后
    if is_main_process():
        torch.save({'vit': vit.state_dict(), 'dec': dec.state_dict()},
                   os.path.join(args.out, 'last_seg.pth'))


if __name__ == '__main__':
    main()

