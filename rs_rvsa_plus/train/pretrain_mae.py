
import os, argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from ..utils.common import set_seed, select_gpus, is_main_process
from ..data.datasets import _gather_images_from_roots
from ..backbones.models_mae import mae_vit_base_patch16

def build_dataset(roots, img_size):
    files = _gather_images_from_roots(roots)
    from PIL import Image
    t = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
    class SimpleImg(torch.utils.data.Dataset):
        def __init__(self, files): self.files = files
        def __len__(self): return len(self.files)
        def __getitem__(self,i):
            x = t(Image.open(self.files[i]).convert('RGB'))
            return x
    return SimpleImg(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default=None, help='e.g., 0,2,3')
    ap.add_argument('--data', nargs='+', required=True, help='roots: path1 path2 path3 ...')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1.5e-4)
    ap.add_argument('--out', default='./outputs_pretrain')
    args = ap.parse_args()

    select_gpus(args.gpus); set_seed(42)
    os.makedirs(args.out, exist_ok=True)

    ds = build_dataset(args.data, args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mae_vit_base_patch16()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for x in dl:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                loss, _, _ = model(x)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()*x.size(0)
        avg = running/len(ds)
        if is_main_process():
            print(f'Epoch {epoch+1}/{args.epochs} loss={avg:.4f}')
            torch.save({'epoch':epoch+1,'model':model.state_dict()}, os.path.join(args.out, f'ckpt_{epoch+1:04d}.pt'))
    torch.save(model.state_dict(), os.path.join(args.out, 'mae_final.pth'))

if __name__=='__main__': main()
