
# import os, argparse
# import torch, torch.nn as nn
# from torch.utils.data import DataLoader
# from ..utils.common import set_seed, select_gpus, is_main_process
# from ..data.datasets import ImageFolderRecursive
# from ..backbones.models_vit import VisionTransformer
# from ..heads.classification import ClsHead

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--gpus', default=None)
#     ap.add_argument('--train', nargs='+', required=True)
#     ap.add_argument('--val', nargs='+', required=True)
#     ap.add_argument('--epochs', type=int, default=30)
#     ap.add_argument('--batch_size', type=int, default=32)
#     ap.add_argument('--img_size', type=int, default=224)
#     ap.add_argument('--lr', type=float, default=3e-4)
#     ap.add_argument('--backbone_ckpt', default=None)
#     ap.add_argument('--train_ratio', type=float, default=0.2)
#     ap.add_argument('--out', default='./outputs_cls')
#     args = ap.parse_args()

#     select_gpus(args.gpus); set_seed(42)
#     os.makedirs(args.out, exist_ok=True)

#     train_ds = ImageFolderRecursive(args.train)
#     val_ds = ImageFolderRecursive(args.val)
#     train_ds.transform.transforms[0].size = (args.img_size,args.img_size)
#     val_ds.transform.transforms[0].size = (args.img_size,args.img_size)
#     dl_tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
#     dl_va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     backbone = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
#     if args.backbone_ckpt and os.path.exists(args.backbone_ckpt):
#         sd = torch.load(args.backbone_ckpt, map_location='cpu')
#         sd = sd.get('model', sd)
#         backbone.load_state_dict(sd, strict=False)
#     dim = getattr(backbone, 'embed_dim', 768)
#     head = ClsHead(dim, num_classes=len(train_ds.class_to_idx))
#     model = torch.nn.Sequential(backbone, torch.nn.Identity(), head).to(device)

#     params = list(backbone.parameters()); n=len(params); k=int(round(n*args.train_ratio))
#     for i,p in enumerate(params):
#         p.requires_grad = i >= (n-k)
#     opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
#     loss_fn = nn.CrossEntropyLoss()

#     best=0.0
#     for epoch in range(args.epochs):
#         model.train()
#         for x,y,_ in dl_tr:
#             x,y = x.to(device), y.to(device)
#             logits = model(x)
#             loss = loss_fn(logits, y)
#             opt.zero_grad(); loss.backward(); opt.step()
#         model.eval(); correct=0; total=0
#         with torch.no_grad():
#             for x,y,_ in dl_va:
#                 x,y = x.to(device), y.to(device)
#                 logits = model(x)
#                 pred = logits.argmax(1)
#                 correct += (pred==y).sum().item(); total += y.numel()
#         acc = correct/total if total>0 else 0
#         if is_main_process():
#             print(f'Epoch {epoch+1}: val acc={acc:.4f}')
#             if acc>best:
#                 best=acc
#                 torch.save({'epoch':epoch+1,'backbone':backbone.state_dict(),'head':head.state_dict()}, os.path.join(args.out,'best_cls.pth'))

# if __name__=='__main__': main()


import os, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.common import set_seed, select_gpus, is_main_process
from ..data.datasets import ImageFolderRecursive
from ..backbones.models_vit import VisionTransformer
from ..heads.classification import ClsHead

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--val', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone_ckpt', default=None)
    ap.add_argument('--train_ratio', type=float, default=0.2)
    ap.add_argument('--out', default='./outputs_cls')
    args = ap.parse_args()

    select_gpus(args.gpus); set_seed(42)
    os.makedirs(args.out, exist_ok=True)

    train_ds = ImageFolderRecursive(args.train)
    val_ds = ImageFolderRecursive(args.val)
    train_ds.transform.transforms[0].size = (args.img_size,args.img_size)
    val_ds.transform.transforms[0].size = (args.img_size,args.img_size)
    dl_tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dl_va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    if args.backbone_ckpt and os.path.exists(args.backbone_ckpt):
        sd = torch.load(args.backbone_ckpt, map_location='cpu')
        sd = sd.get('model', sd)
        vit.load_state_dict(sd, strict=False)
    dim = getattr(vit, 'embed_dim', 768)
    head = ClsHead(dim, num_classes=len(train_ds.class_to_idx))

    vit.to(device); head.to(device)

    # 部分解冻：仅解冻最后 train_ratio 比例的参数
    params = list(vit.parameters()); n = len(params); k = int(round(n * args.train_ratio))
    for i, p in enumerate(params):
        p.requires_grad = i >= (n - k)

    opt = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, vit.parameters())) + list(head.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(args.epochs):
        vit.train(); head.train()
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            # 关键：直接用 forward_features，避免 attn_mask 被透传
            feats = vit.forward_features(x)
            if feats.ndim == 3:  # 取 CLS token
                feats = feats[:, 0]
            logits = head(feats)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

        vit.eval(); head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, _ in dl_va:
                x, y = x.to(device), y.to(device)
                feats = vit.forward_features(x)
                if feats.ndim == 3:
                    feats = feats[:, 0]
                pred = head(feats).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / total if total > 0 else 0.0

        if is_main_process():
            print(f'Epoch {epoch+1}: val acc={acc:.4f}')
            if acc > best:
                best = acc
                torch.save({'epoch': epoch+1, 'backbone': vit.state_dict(), 'head': head.state_dict()},
                           os.path.join(args.out, 'best_cls.pth'))

if __name__ == '__main__':
    main()
