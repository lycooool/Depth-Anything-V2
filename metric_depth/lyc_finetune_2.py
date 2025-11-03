import argparse
import logging
import os
import pprint
import random
from tqdm import tqdm
import sys
# 讓 Python 能找到上層路徑的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# from dataset.hypersim import Hypersim
# from dataset.kitti import KITTI
# from dataset.vkitti2 import VKITTI2
from dataset.LayeredDepth_Syn import LayeredDepth_Syn
from depth_anything_v2.dpt_finetune_2 import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
#parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=30, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=4, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', default='/home/lyc/research/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth', type=str)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    
    #warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    # rank, world_size = setup_distributed(port=args.port) # 非分佈式訓練 不需要
    rank, world_size = 0, 1
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)

    trainset = LayeredDepth_Syn(mode = 'train')
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=True)

    valset = LayeredDepth_Syn(mode = 'validation')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)
    
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    #model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model = DepthAnythingV2(**{**model_configs[args.encoder]})
    
    if args.pretrained_from:
        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = SiLogLoss().to(DEVICE)
    
    # pretrained 的地方 learning rate 調低
    optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                       {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                      lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)


   
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

    
    for epoch in range(args.epochs):

        if rank == 0:
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch}/{args.epochs} — Previous Best Results")
            logger.info("=" * 80)
            logger.info(
                ", ".join([f"{k}: {previous_best[k]:.3f}" for k in previous_best.keys()])
            )
            logger.info("=" * 80)
        
        #trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()

        # ✅ tqdm 進度條
        progress_bar = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            ncols=100,
            dynamic_ncols=True
        )
        ######################################
        for i, sample in progress_bar:
            # if i == 1:
            #     break
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].cuda(), sample['d1'].cuda(), sample['d1_valid_mask'].cuda()

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            
            pred = model(img)
            
            

            
            
            loss = criterion(pred, depth, (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth))
            
            loss.backward()
            optimizer.step()
            
            
            # total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9

            
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            progress_bar.set_postfix({
                'L1': f"{loss.item():.3f}",
                'LR': f"{lr:.2e}"
            })
            
            if rank == 0:
                #writer.add_scalar('train/loss', loss.item(), iters)
                writer.add_scalar('train/loss_l1', loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(
                    f"Iter: {i}/{len(trainloader)}, "
                    f"LR(backbone): {lr:.7f}, LR(heads): {lr*10:.7f}, "
                    f"L1: {loss.item():.3f}"
                )
            
        model.eval()


        # ✅ 加入進度條顯示
        progress_bar_val = tqdm(
            enumerate(valloader),
            total=len(valloader),
            desc=f"[Eval] Epoch {epoch+1}/{args.epochs}",
            ncols=100,
            dynamic_ncols=True
        )

        
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}

        nsamples = torch.tensor([0.0]).cuda()

        
        
        for i, sample in progress_bar_val:
    
            
            img, depth, valid_mask = sample['image'].cuda(), sample['d1'].cuda(), sample['d1_valid_mask'].cuda()[0]

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

            depth = depth.squeeze()             # [1,H,W] → [H,W]
            valid_mask = valid_mask.squeeze()
            valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

            if valid_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

            
        
        if rank == 0:  # 主 GPU 或單 GPU
            logger.info("=" * 100)
            logger.info("---- Evaluation Results ----")
            logger.info(
                "{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(*tuple(results.keys()))
            )
            logger.info(
                "{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(
                    *tuple([(v / nsamples).item() for v in results.values()])
                )
            )
            logger.info("=" * 100)

            # 寫入 TensorBoard
            for name, metric in results.items():
                avg_value = (metric / nsamples).item()
                writer.add_scalar(f"eval/{name}", avg_value, epoch)

        # === 更新最佳結果 (previous_best)
        for k in results.keys():
            avg_value = (results[k] / nsamples).item()

            if k in ["d1", "d2", "d3"]:  # 越大越好
                previous_best[k] = max(previous_best[k], avg_value)
            else:                        # 越小越好
                previous_best[k] = min(previous_best[k], avg_value)

        # === 儲存 checkpoint ===
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, f"latest_epoch{epoch}.pth"))


if __name__ == '__main__':
    main()