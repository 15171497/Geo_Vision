
import os, glob
from typing import List, Optional, Callable
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def _gather_images_from_roots(roots: List[str], exts=('.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp')):
    files = []
    for r in roots:
        for e in exts:
            files += glob.glob(os.path.join(r, '**', f'*{e}'), recursive=True)
    files = sorted({f for f in files if os.path.isfile(f)})
    return files

class ImageFolderRecursive(Dataset):
    def __init__(self, roots: List[str], transform: Optional[Callable]=None):
        self.samples = []
        self.class_to_idx = {}
        class_dirs = []
        for r in roots:
            for dirpath, dirnames, filenames in os.walk(r):
                if any(fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')) for fn in filenames):
                    parts = os.path.relpath(dirpath, r).split(os.sep)
                    if len(parts)>=1 and parts[0] not in ('.',''):
                        class_dirs.append((r, parts[0], dirpath))
        classes = sorted({c for _,c,_ in class_dirs})
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for r,c,dirpath in class_dirs:
            idx = self.class_to_idx[c]
            for fn in glob.glob(os.path.join(dirpath, '**', '*.*'), recursive=True):
                if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')):
                    self.samples.append((fn, idx))
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor()])

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        return x, y, path

class SegmentationDataset(Dataset):
    def __init__(self, root_or_list: str, img_size: int=512, num_classes: int=2, transform: Optional[Callable]=None):
        pairs = []
        if os.path.isfile(root_or_list):
            with open(root_or_list,'r',encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    ip, mp = line.split()
                    pairs.append((ip, mp))
        else:
            img_root = os.path.join(root_or_list, 'images')
            mask_root = os.path.join(root_or_list, 'masks')
            for ip in _gather_images_from_roots([img_root]):
                rel = os.path.relpath(ip, img_root)
                mp = os.path.join(mask_root, os.path.splitext(rel)[0] + '.png')
                if os.path.exists(mp):
                    pairs.append((ip, mp))
        self.pairs = pairs
        self.num_classes = num_classes
        self.transform_img = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
        self.transform_mask = T.Compose([T.Resize((img_size,img_size), interpolation=T.InterpolationMode.NEAREST)])

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert('RGB')
        mask = Image.open(mp)
        x = self.transform_img(img)
        y = self.transform_mask(mask)
        y = (T.PILToTensor()(y).squeeze(0)).long()
        return x, y, ip

class ChangeDetectionDataset(Dataset):
    def __init__(self, list_txt: str, img_size: int=512, transform: Optional[Callable]=None):
        triplets = []
        with open(list_txt,'r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                a,b,l = line.split()
                triplets.append((a,b,l))
        self.triplets = triplets
        self.t_img = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
        self.t_lbl = T.Compose([T.Resize((img_size,img_size), interpolation=T.InterpolationMode.NEAREST)])

    def __len__(self): return len(self.triplets)
    def __getitem__(self, i):
        a,b,l = self.triplets[i]
        A = self.t_img(Image.open(a).convert('RGB'))
        B = self.t_img(Image.open(b).convert('RGB'))
        L = self.t_lbl(Image.open(l))
        L = (T.PILToTensor()(L).squeeze(0)>0).long()
        x = (A,B)
        return x, L, (a,b,l)
