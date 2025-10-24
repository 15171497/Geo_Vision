
import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.common import set_seed, select_gpus, is_main_process
from ..data.datasets import ChangeDetectionDataset
from ..backbones.models_vit import VisionTransformer
from ..heads.change_detection import SiameseChangeDecoder

class ViTEncoder2D(nn.Module):
    def __init__(self, vit: VisionTransformer, img_size: int, patch: int=16):
        super().__init__()
        self.vit = vit; self.img_size=img_size; self.patch=patch
    def forward(self, x):
        B = x.size(0)
        feats = self.vit.forward_features(x)
        C = feats.size(-1)
        n = (self.img_size//self.patch)**2
        fmap = feats[:,1:1+n,:].transpose(1,2).reshape(B, C, self.img_size//self.patch, self.img_size//self.patch)
        return fmap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--list', required=True, help='txt: A B label')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone_ckpt', default=None)
    ap.add_argument('--train_ratio', type=float, default=0.2)
    ap.add_argument('--out', default='./outputs_cd')
    args = ap.parse_args()

    select_gpus(args.gpus); set_seed(42)
    os.makedirs(args.out, exist_ok=True)

    ds = ChangeDetectionDataset(args.list, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vitA = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    vitB = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    if args.backbone_ckpt and os.path.exists(args.backbone_ckpt):
        sd = torch.load(args.backbone_ckpt, map_location='cpu')
        sd = sd.get('model', sd)
        vitA.load_state_dict(sd, strict=False); vitB.load_state_dict(sd, strict=False)
    encA = ViTEncoder2D(vitA, img_size=args.img_size)
    encB = ViTEncoder2D(vitB, img_size=args.img_size)
    dim = getattr(vitA,'embed_dim',768)
    dec = SiameseChangeDecoder(embed_dim=dim)
    model = nn.Module(); model.encA=encA; model.encB=encB; model.dec=dec
    model.to(device)

    params = list(vitA.parameters())+list(vitB.parameters()); n=len(params); k=int(round(n*args.train_ratio))
    for i,p in enumerate(params):
        p.requires_grad = i >= (n-k)
    opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train(); running=0
        for (A,B),L,_ in dl:
            A,B,L = A.to(device), B.to(device), L.to(device).float()
            fA = model.encA(A); fB = model.encB(B)
            logits = model.dec(fA,fB).squeeze(1)
            loss = loss_fn(logits, L)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()*A.size(0)
        avg = running/len(ds)
        if is_main_process():
            print(f'Epoch {epoch+1}: loss={avg:.4f}')

    if is_main_process():
        torch.save({'vitA':vitA.state_dict(),'vitB':vitB.state_dict(),'dec':dec.state_dict()}, os.path.join(args.out,'last_cd.pth'))

if __name__=='__main__': main()
