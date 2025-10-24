# GeoVision — Remote Sensing Foundation + Finetuning Suite
希望大家给我点个star哦
> 预训练 MAE + ViT/RVSA 骨干，支持分类 / 语义分割 / 变化检测微调与推理。  
> 已修复并增强：`pos_embed` 尺寸不匹配、`attn_mask` 兼容、忽略像素（255/0）、输出尺寸自动对齐、健壮的 ViT token 提取、批量验证集推理（含彩色可视化）、逐类 mIoU 评估。

---

## ✨ 主要特性

- **架构**：使用RVSA/ViT/MAE 主干；
- **MAE 预训练**：递归收集任意路径下的遥感影像，自监督像素重建目标；
- **三大任务微调**  
  - **分类**：CLS 线性头，兼容 timm 不同版本；  
  - **语义分割**：ViT 编码 + 轻量解码器（可任意图像大小，自动对齐标签尺寸）；  
  - **变化检测**：Siamese ViT + |A−B| 解码；
- **稳健加载**：自动对 `pos_embed` 做 2D 双三次插值（例如预训练 224 → 微调 512/1024）；  
- **忽略像素**：分割任务可将 **255 或 0** 标为忽略（`ignore_index`）；  
- **评估指标**：像素精度（PixelAcc）+ **逐类 IoU → mIoU**（而非“总体 IoU 近似”）；  
- **推理工具**：  
  - 单图推理；  
  - **批量验证集推理**（读 `val_list.txt`），导出**灰度掩码**与**彩色可视化**，并把**原图**复制到输出目录；  
- **多 GPU**：  
  - Fixed 版支持用 `--gpus` 精确选择 GPU（单进程）；  

---

## 🗂️ 目录结构（核心）
完整的训练好的模型（预训练，微调），在百度网盘中：通过网盘分享的文件：GeoVista
链接: https://pan.baidu.com/s/1QN91AfpJSV1jPxLLkRrPFg?pwd=em2t 提取码: em2t
```
rs_rvsa_plus/
  backbones/
    models_mae.py            # MAE
    models_vit.py            # ViT / RVSA 兼容
    util/pos_embed.py        # 正余弦位置编码
  heads/
    classification.py        # 分类头
    segmentation.py          # 分割解码器
    change_detection.py      # 变化检测解码器
  data/
    datasets.py              # 递归图像收集、分割/变化检测数据集
  train/
    pretrain_mae.py          # MAE 预训练
    finetune_cls.py          # 分类微调（attn_mask 安全）
    finetune_seg.py          # 分割微调（健壮 token 提取 + 尺寸对齐 + ignore_index + mIoU）
    finetune_cd.py           # 变化检测微调
  infer/
    infer_seg.py             # 单图分割推理（已对齐训练逻辑）
    infer_seg_batch.py       # ★ 批量验证集分割推理 + 彩色可视化 + 原图复制
    copy_val_originals.py    # 仅复制验证集原图的小工具
  utils/
    common.py                # 随机种子 / 选择 GPU / 主进程判断
requirements.txt
README.md  (本文)
```

---

## 🧰 环境

与之前保持一致（示例）：

```
torch>=2.0
torchvision>=0.15
timm>=0.9.2
numpy
Pillow
opencv-python
```

> 若 timm 版本不同也可用；代码已对 `forward()/forward_features()` 兼容处理。

---

## 🧪 数据组织

### 预训练（自监督）

- 传入多个任意根路径：`path1 path2 path3 ...`，在每个根目录**递归搜索**所有影像 (`.jpg/.png/.tif/.bmp/.webp` 等)。

### 分类

```
/train_root/
  class_A/**.png|jpg|tif...
  class_B/**...
/val_root/
  class_A/**...
  class_B/**...
```

类别即一级子目录名。

### 语义分割（两种方式）

**A. 推荐**：`root/images/**` 与 `root/masks/**`，同相对路径 + 掩码为 `.png`  
**B. list 文件**：每行 `image_path mask_path`（绝对或相对路径都可）

> 掩码为**单通道整数类别图** `[0..K-1]`，未标注可用 **255**（或 LoveDA 的 **0**）并设置 `--ignore_index`。

### 变化检测

`list.txt`：每行 `imgA imgB label`（label 可 0/1 或 0/255；代码中统一 `>0` 为 1）

---

## 🚀 训练

### 1) 预训练（MAE）

```bash
python -m rs_rvsa_plus.train.pretrain_mae \
  --gpus 0 \
  --data /data/pretrain/path1 /data/pretrain/path2 \
  --img_size 224 --epochs 100 --batch_size 128 \
  --out outputs_pretrain
```

### 2) 分类微调

```bash
python -m rs_rvsa_plus.train.finetune_cls \
  --gpus 0 \
  --train /data/cls/train_root \
  --val   /data/cls/val_root \
  --img_size 224 --epochs 30 --batch_size 64 \
  --backbone_ckpt outputs_pretrain/mae_final.pth \
  --train_ratio 0.2 \
  --out outputs_cls
```

### 3) 分割微调（支持任意输入大小 + 自动对齐标签）

```bash
python -m rs_rvsa_plus.train.finetune_seg \
  --gpus 0 \
  --train /data/seg/train_root \
  --val   /data/seg/val_root \
  --num_classes 7 --img_size 512 --epochs 80 --batch_size 4 \
  --backbone_ckpt outputs_pretrain/mae_final.pth \
  --train_ratio 0.2 \
  --ignore_index 255 \
  --out outputs_seg
```

> LoveDA 如将 **0** 视为无效像素，则改 `--ignore_index 0`。

### 4) 变化检测微调

```bash
python -m rs_rvsa_plus.train.finetune_cd \
  --gpus 0 \
  --list /data/cd/train_list.txt \
  --img_size 512 --epochs 60 --batch_size 4 \
  --backbone_ckpt outputs_pretrain/mae_final.pth \
  --train_ratio 0.2 \
  --out outputs_cd
```

---

## 🔍 推理 / 可视化

### 单图分割推理

展示一下语义分割结果与真实图像的对比。

![](file:///D:/file/python_file5/GeoVista/image/all_pred.png)

```bash
python -m rs_rvsa_plus.infer.infer_seg \
  --gpus 0 \
  --img /path/to/image.png \
  --ckpt outputs_seg/best_seg.pth \
  --img_size 512 \
  --num_classes 7 \
  --restore_original_size \
  --out seg_pred.png
```

### 批量验证集分割推理（灰度 + 彩色 + 标签彩色 + 复制原图）

- `val_list.txt` 格式：`<image_path> <mask_path>`
  
  ```bash
  python -m rs_rvsa_plus.infer.infer_seg_batch \
  --gpus 0 \
  --list /mnt/public/lyb/Dataset/Semantic_Segmentation/LOVEDA/val_list.txt \
  --ckpt outputs_seg/best_seg.pth \
  --img_size 512 \
  --num_classes 7 \
  --ignore_index -1 \
  --out_dir outputs_seg_val_vis \
  --restore_original_size
  ```

输出结构示例：

```
outputs_seg_val_vis/
  pred_gray/       # 预测灰度掩码（类别 id）
  pred_color/      # 预测彩色可视化
  label_color/     # 标签彩色可视化
  originals/       # 验证集原图拷贝
```

> 如需仅复制原图：`python -m rs_rvsa_plus.infer.copy_val_originals --list val_list.txt --out_dir outputs_seg_val_vis`

---

## 📊 指标计算

训练脚本中评估函数为**逐类 IoU → mIoU**（忽略 `ignore_index` 像素）：

- **PixelAcc**：正确像素 / 有效像素；
- **mIoU**：对出现过的每个类别分别计算 IoU 后平均。

---

## 🧠 关键实现与设计说明

- **pos_embed 插值**：  
  - 预训练 224（14×14 token） → 微调 512（32×32 token）会触发**位置编码尺寸不匹配**；  
  - 我们在 **加载 checkpoint** 与 **手工 token 流**中都对 `pos_embed` 做了 2D 双三次插值，保证无缝迁移。
- **健壮 ViT 编码器（分割）**：  
  - 某些 timm 版本的 `forward_features` 只返回 `[B,C]`（仅 CLS）；  
  - 代码会尝试 `return_all_tokens=True`；若失败则**改走手工路径**（patch_embed + pos + blocks + norm），稳定得到 `[B,1+N,C]`；
- **输出尺寸自动对齐**：  
  - 解码器默认上采样到 `S/2`；训练与推理前统一将 `logits` **插值对齐**到目标标签或 `img_size`；
- **忽略像素**：  
  - 分割训练使用 `CrossEntropyLoss(ignore_index=...)`，默认 255；LoveDA 可设 0；  
  - 评估与可视化同样会跳过忽略像素（或以特定颜色渲染）。

---

## 🧩 常见问题（FAQ）

- **`pos_embed` size mismatch**  
  使用本仓脚本会自动插值（无需改模型）；确保你传入了正确的 `--img_size`。
- **`attn_mask` unexpected in forward_features**  
  分类脚本已改用 `forward_features + 线性头` 的安全路径，不再透传 `attn_mask`。
- **`input and target sizes don't match`（分割）**  
  已在训练/评估中对 `logits` 做输出尺寸对齐；如仍出现，请检查你的标签与 `--img_size` 是否期望一致。
- **LoveDA 的无效像素**  
  LoveDA 常用 **0** 为无效像素；请在分割训练/可视化时设 `--ignore_index 0`。

---

## 🛠️ 可调参数（常用）

- `--img_size`：任意（建议 224/512/1024 等，显存允许尽量大）；  
- `--train_ratio`：ViT 解冻比例（0~1）；数据少时小一些，数据多可加大；  
- `--ignore_index`：未标注像素标签值（255 或 0 等）；  
- `--gpus`：逗号分隔的 GPU 列表（如 `--gpus 0,1`）；  
- `--batch_size` / `--epochs` / `--lr`：按资源与数据规模调整。

---



## 📜 许可与致谢

- 如果需要学习交流，或者想要数据集或者训练好的模型，可以联系lyb2455716@163.com。
- Backbone/MAE 参考原始 RVSA/ViT/MAE 实现，保留其核心设计与接口；  
- 工作版权归本人所有。
