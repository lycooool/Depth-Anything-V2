from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch
import numpy as np
import cv2

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from datasets import load_dataset


class LayeredDepth_Syn(Dataset):
    def __init__(self, mode='train', size=(518, 518), use_layers=(1, 3, 5, 7)):
        """
        mode: 'train' 或 'val'，決定使用的 dataset split
        use_layers: 要使用哪些層的深度，例如 (1,3,5,7)
        """
        # === 1️⃣ 根據 mode 決定 split ===
        if mode == 'train':
            split = 'train'
        elif mode == 'validation':
            split = 'validation'


        # === 2️⃣ 載入對應的 Hugging Face dataset split ===
        self.dataset = load_dataset("princeton-vl/LayeredDepth-Syn", split=split)
        self.mode = mode
        self.size = size
        self.use_layers = use_layers
        
        # === 3️⃣ 建立 transform pipeline（與 Hypersim 相同邏輯）===
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))  # ✅ 訓練才裁切

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # === 1️⃣ 讀取影像 ===
        image = np.array(sample['image.png']).astype(np.float32) / 255.0
        
        # === 2️⃣ 根據 use_layers 取出多層深度 ===
        depth_layers = []
        for layer_id in self.use_layers:
            key = f'depth_{layer_id}.png'
            if key not in sample:
                raise KeyError(f"{key} not found in dataset sample. Available keys: {list(sample.keys())}")
            
            depth_pil = sample[key]
            # ⚠️ 保留 16-bit：轉 numpy.uint16
            depth_map = np.array(depth_pil, dtype=np.uint16).astype(np.float32)
            depth_map = np.array(sample[key]).astype(np.float32)
            depth_layers.append(depth_map)
        
        # shape: [H, W, N_layers]
        depth = np.stack(depth_layers, axis=-1)

        # === 3️⃣ 套用 Transform（resize + normalize + prepare）===
        out = self.transform({'image': image, 'depth': depth})
        
        # === 4️⃣ 轉成 Tensor ===
        
        out['image'] = torch.from_numpy(out['image'])  # [3,H,W]

        # ✅ 改成 [N_layers, H, W]，和模型輸出對齊
        depth_tensor = torch.from_numpy(out['depth']).permute(2, 0, 1).contiguous().float()

        # === 5️⃣ 建立 mask ===
        valid_mask = (depth_tensor > 0).float()

        # 每個輸出分開
        for i, layer_id in enumerate(self.use_layers):
            out[f'd{layer_id}'] = depth_tensor[i]
            out[f'd{layer_id}_valid_mask'] = valid_mask[i]

        
        
        return out
        
    def __len__(self):
        return len(self.dataset)
    
    