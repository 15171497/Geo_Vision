
import os, argparse, torch
from ..data.datasets import ImageFolderRecursive
from ..backbones.models_vit import VisionTransformer
from ..heads.classification import ClsHead

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=224)
    args = ap.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = ImageFolderRecursive(args.data)
    ds.transform.transforms[0].size = (args.img_size,args.img_size)
    vit = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    head = ClsHead(getattr(vit,'embed_dim',768), num_classes=len(ds.class_to_idx))
    sd = torch.load(args.ckpt, map_location='cpu')
    vit.load_state_dict(sd.get('backbone',{}), strict=False); head.load_state_dict(sd.get('head',{}), strict=False)
    vit.to(device); head.to(device); vit.eval(); head.eval()
    with torch.no_grad():
        for i in range(len(ds)):
        #for i in range(min(10, len(ds))):
            x,_,path = ds[i]
            x = x.unsqueeze(0).to(device)
            feats = vit.forward_features(x)
            if feats.ndim==3: feats = feats[:,0]
            logits = head(feats)
            pred = logits.argmax(1).item()
            print(path, pred)
if __name__=='__main__': main()
