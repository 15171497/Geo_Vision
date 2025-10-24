
import os, argparse, torch
from PIL import Image
from ..backbones.models_vit import VisionTransformer
from ..train.finetune_cd import ViTEncoder2D
from ..heads.change_detection import SiameseChangeDecoder
import torchvision.transforms as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None)
    ap.add_argument('--imgA', required=True)
    ap.add_argument('--imgB', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--out', default='cd_pred.png')
    args = ap.parse_args()

    if args.gpus is not None: os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vitA = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    vitB = VisionTransformer(img_size=args.img_size, patch_size=16, in_chans=3, num_classes=0)
    encA = ViTEncoder2D(vitA, img_size=args.img_size)
    encB = ViTEncoder2D(vitB, img_size=args.img_size)
    dec = SiameseChangeDecoder(embed_dim=getattr(vitA,'embed_dim',768))
    sd = torch.load(args.ckpt, map_location='cpu')
    vitA.load_state_dict(sd.get('vitA',{}), strict=False); vitB.load_state_dict(sd.get('vitB',{}), strict=False); dec.load_state_dict(sd.get('dec',{}), strict=False)
    vitA.to(device); vitB.to(device); dec.to(device)
    vitA.eval(); vitB.eval(); dec.eval()

    t = T.Compose([T.Resize((args.img_size,args.img_size)), T.ToTensor()])
    A = t(Image.open(args.imgA).convert('RGB')).unsqueeze(0).to(device)
    B = t(Image.open(args.imgB).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        fA = encA(A); fB = encB(B)
        logits = dec(fA,fB).squeeze(1)
        pred = (torch.sigmoid(logits)>0.5).squeeze(0).cpu().byte()*255
        Image.fromarray(pred.numpy()).save(args.out)
        print('Saved', args.out)

if __name__=='__main__': main()
