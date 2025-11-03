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
from depth_anything_v2.dpt_finetune_1 import DepthAnythingV2
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
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True)

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
        # ckpt = torch.load(args.pretrained_from, map_location='cpu')

        # # 篩出包含 'pretrained' 的 key
        # pretrained_keys = [k for k in ckpt.keys() if 'pretrained' in k]

        # print("\n=== ✅ Loaded pretrained layers ===")
        # for k in pretrained_keys:
        #     print(k)
        # print(f"Total pretrained layers loaded: {len(pretrained_keys)}")

        model.load_state_dict({k: v for k, v in torch.load(args.pretrained_from, map_location='cpu').items() if 'pretrained' in k}, strict=False)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = SiLogLoss().to(DEVICE)
    
    # pretrained 的地方 learning rate 調低
    # optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
    #                    {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
    #                   lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)


    for param in model.pretrained.parameters():
        param.requires_grad = False
    
    optim_l1 = AdamW(model.depth_heads[0].parameters(), lr=args.lr*10)
    optim_l3 = AdamW(model.depth_heads[1].parameters(), lr=args.lr*10)
    optim_l5 = AdamW(model.depth_heads[2].parameters(), lr=args.lr*10)
    optim_l7 = AdamW(model.depth_heads[3].parameters(), lr=args.lr*10)
    optim_backbone = AdamW([p for n,p in model.named_parameters() if 'pretrained' in n], lr=args.lr)
    
    total_iters = args.epochs * len(trainloader)
    
    #previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    # 定義 head 名稱
    head_names = ['l1', 'l3', 'l5', 'l7']

    # 定義要追蹤的評估指標
    metrics = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']

    # 建立巢狀字典形式的 previous_best
    previous_best = {
        h: {m: (0.0 if m in ['d1', 'd2', 'd3'] else 100.0) for m in metrics}
        for h in head_names
    }
    
    for epoch in range(args.epochs):
        # if rank == 0:
        #     logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
        #     logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
        #                 'log10: {:.3f}, silog: {:.3f}'.format(
        #                     epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
        #                     previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        if rank == 0:
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch}/{args.epochs} — Previous Best Results by Head")
            logger.info("=" * 80)
            for h in head_names:
                best_vals = previous_best[h]
                logger.info(
                    f"[{h}] " +
                    ", ".join([f"{k}: {best_vals[k]:.3f}" for k in metrics])
                )
            logger.info("=" * 80)
        
        #trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0

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

            # if i == 10:
            #     break
            

            # if i == 0:
            #     print("🔍 sample keys:", sample.keys())
            #     for k, v in sample.items():
            #         if torch.is_tensor(v):
            #             print(f"{k}: {v.shape}, dtype={v.dtype}")
            #         else:
            #             print(f"{k}: {type(v)}")
            # #optimizer.zero_grad()
            
            #img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()

            # if random.random() < 0.5:
            #     img = img.flip(-1)
            #     depth = depth.flip(-1)
            #     valid_mask = valid_mask.flip(-1)

            optim_l1.zero_grad()
            optim_l3.zero_grad()
            optim_l5.zero_grad()
            optim_l7.zero_grad()
            optim_backbone.zero_grad()

            img = sample['image'].cuda()
            # 輸出 img 的形狀
            # print("shape of img:")
            # print(img.shape)
            depth_l1, valid_mask_l1 = sample['d1'].cuda(), sample['d1_valid_mask'].cuda()
            depth_l3, valid_mask_l3 = sample['d3'].cuda(), sample['d3_valid_mask'].cuda()
            depth_l5, valid_mask_l5 = sample['d5'].cuda(), sample['d5_valid_mask'].cuda()
            depth_l7, valid_mask_l7 = sample['d7'].cuda(), sample['d7_valid_mask'].cuda()
            # 轉成公尺
            # depth_l1 = depth_l1 / 1000.0
            # depth_l3 = depth_l3 / 1000.0
            # depth_l5 = depth_l5 / 1000.0
            # depth_l7 = depth_l7 / 1000.0

            if random.random() < 0.5:
                img = img.flip(-1)
                depth_l1, depth_l3, depth_l5, depth_l7 = (d.flip(-1) for d in [depth_l1, depth_l3, depth_l5, depth_l7])
                valid_mask_l1, valid_mask_l3, valid_mask_l5, valid_mask_l7 = (d.flip(-1) for d in [valid_mask_l1, valid_mask_l3, valid_mask_l5, valid_mask_l7])
            
            pred = model(img)
            
            min_depth = torch.tensor(args.min_depth, device="cuda")
            max_depth = torch.tensor(args.max_depth, device="cuda")
            
            # #=== Debug: 檢查所有 head 的預測與真值範圍 ===
            # for idx, (p, d, m, name) in enumerate(zip(
            #     pred, 
            #     [depth_l1, depth_l3, depth_l5, depth_l7],
            #     [valid_mask_l1, valid_mask_l3, valid_mask_l5, valid_mask_l7],
            #     ["l1", "l3", "l5", "l7"]
            # )):
            #     # 安全轉換成 float
            #     p_min, p_max = p.min().item(), p.max().item()
            #     d_min, d_max = d.min().item(), d.max().item()
            #     mask_ratio = m.sum().item() / m.numel()

            #     print(f"[{name}] pred min={p_min:.6f}, max={p_max:.6f} | "
            #         f"depth min={d_min:.6f}, max={d_max:.6f} | "
            #         f"mask ratio={mask_ratio:.3f}")

            loss_l1 = criterion(pred[0], depth_l1, (valid_mask_l1 == 1) & (depth_l1 >= min_depth) & (depth_l1 <= max_depth))
            loss_l3 = criterion(pred[1], depth_l3, (valid_mask_l3 == 1) & (depth_l3 >= min_depth) & (depth_l3 <= max_depth))
            loss_l5 = criterion(pred[2], depth_l5, (valid_mask_l5 == 1) & (depth_l5 >= min_depth) & (depth_l5 <= max_depth))
            loss_l7 = criterion(pred[3], depth_l7, (valid_mask_l7 == 1) & (depth_l7 >= min_depth) & (depth_l7 <= max_depth))
            # print(f"[DEBUG] L1 loss: {loss_l1.item():.6f}, "
            #     f"L3 loss: {loss_l3.item():.6f}, "
            #     f"L5 loss: {loss_l5.item():.6f}, "
            #     f"L7 loss: {loss_l7.item():.6f}")

            #每個 layer 分開傳
            # Head 1
            loss_l1.backward(retain_graph=True)   # 保留計算圖給其他 head
            optim_l1.step()

            # Head 3
            loss_l3.backward(retain_graph=True)
            optim_l3.step()

            # Head 5
            loss_l5.backward(retain_graph=True)
            optim_l5.step()

            # Head 7（最後一個，不需 retain_graph）
            loss_l7.backward()
            optim_l7.step()

            # ===============================
            # 3️⃣ Backbone 更新（根據所有 head 的梯度）
            # ===============================
            optim_backbone.step()

            # # 整合各個 head 再傳
            # total_loss = loss_l1 + loss_l3 + loss_l5 + loss_l7
            # total_loss.backward()          # 🔁 一次反向傳遞整個圖
            # optim_backbone.step()          # ✅ 一次更新 backbone
            # for opt in [optim_l1, optim_l3, optim_l5, optim_l7]:
            #     opt.step()                 # ✅ 各 head 一起更新

            
            
            # loss.backward()
            # optimizer.step()
            
            # total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9

            # 動態更新各 optimizer 的學習率
            for param_group in optim_backbone.param_groups:
                param_group["lr"] = lr                # backbone: 原始 lr
            for opt in [optim_l1, optim_l3, optim_l5, optim_l7]:
                for param_group in opt.param_groups:
                    param_group["lr"] = lr * 10.0     # heads: lr × 10
            
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * 10.0

            progress_bar.set_postfix({
                'L1': f"{loss_l1.item():.3f}",
                'L3': f"{loss_l3.item():.3f}",
                'L5': f"{loss_l5.item():.3f}",
                'L7': f"{loss_l7.item():.3f}",
                'LR': f"{lr:.2e}"
            })
            
            if rank == 0:
                #writer.add_scalar('train/loss', loss.item(), iters)
                writer.add_scalar('train/loss_l1', loss_l1.item(), iters)
                writer.add_scalar('train/loss_l3', loss_l3.item(), iters)
                writer.add_scalar('train/loss_l5', loss_l5.item(), iters)
                writer.add_scalar('train/loss_l7', loss_l7.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(
                    f"Iter: {i}/{len(trainloader)}, "
                    f"LR(backbone): {lr:.7f}, LR(heads): {lr*10:.7f}, "
                    f"L1: {loss_l1.item():.3f}, L3: {loss_l3.item():.3f}, "
                    f"L5: {loss_l5.item():.3f}, L7: {loss_l7.item():.3f}"
                )
            ###############################
            # if rank == 0 and i % 100 == 0:
            #     logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))

        model.eval()


        # ✅ 加入進度條顯示
        progress_bar_val = tqdm(
            enumerate(valloader),
            total=len(valloader),
            desc=f"[Eval] Epoch {epoch+1}/{args.epochs}",
            ncols=100,
            dynamic_ncols=True
        )

        
        # results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
        #            'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
        #            'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}

        # nsamples = torch.tensor([0.0]).cuda()

        # 定義 head 名稱
        head_names = ['l1', 'l3', 'l5', 'l7']

        # 定義要追蹤的評估指標
        metrics = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']

        # 建立巢狀字典
        results = {
            h: {m: torch.tensor([0.0]).cuda() for m in metrics}
            for h in head_names
        }

        # 每個 head 也需要紀錄樣本數
        nsamples = {h: torch.tensor([0.0]).cuda() for h in head_names}
        
        
        for i, sample in progress_bar_val:
    
            
            # img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]
            img = sample['image'].cuda().float()

            depth_l1, valid_mask_l1 = sample['d1'].cuda(), sample['d1_valid_mask'].cuda()
            depth_l3, valid_mask_l3 = sample['d3'].cuda(), sample['d3_valid_mask'].cuda()
            depth_l5, valid_mask_l5 = sample['d5'].cuda(), sample['d5_valid_mask'].cuda()
            depth_l7, valid_mask_l7 = sample['d7'].cuda(), sample['d7_valid_mask'].cuda()
            # 轉成公尺
            # depth_l1 = depth_l1 / 1000.0
            # depth_l3 = depth_l3 / 1000.0
            # depth_l5 = depth_l5 / 1000.0
            # depth_l7 = depth_l7 / 1000.0

            with torch.no_grad():
                preds = model(img)
                # for i, p in enumerate(preds):
                #     print(f"Head {i}: {p.shape}")
                # pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
                preds = [F.interpolate(p[:, None], depth_l1.shape[-2:], mode='bilinear', align_corners=True)[0, 0] for p in preds]
                # for i, p in enumerate(preds):
                #     print(f"Head {i}: {p.shape}")
            
            # valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            # if valid_mask.sum() < 10:
            #     continue
            
            # cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            
            # for k in results.keys():
            #     results[k] += cur_results[k]
            # nsamples += 1

            for h, pred, depth, mask in zip(
                head_names,
                preds,                             # 4 個預測輸出
                [depth_l1, depth_l3, depth_l5, depth_l7],
                [valid_mask_l1, valid_mask_l3, valid_mask_l5, valid_mask_l7]
            ):
                valid_mask = (mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
                depth = depth.squeeze(0)
                valid_mask = valid_mask.squeeze(0)
                valid_count = valid_mask.sum().item()
                total_count = valid_mask.numel()
                ratio = valid_count / total_count

                # 🔍 印出 mask 狀況與範圍
                # print(
                #     f"[{h}] valid pixels = {valid_count}/{total_count} "
                #     f"({ratio*100:.2f}%) | "
                #     f"depth range=({depth.min().item():.3f}, {depth.max().item():.3f}) | "
                #     f"pred range=({pred.min().item():.3f}, {pred.max().item():.3f})",
                #     flush=True
                # )

                if valid_mask.sum() < 10:
                    continue
                
                # print(pred.shape)
                # print(depth.shape)
                # print(valid_mask.shape)
                cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
                # print(f"[{h}] cur_results:", cur_results)

                for k in results[h].keys():
                    results[h][k] += cur_results[k]
                nsamples[h] += 1
        
        # torch.distributed.barrier() # 多 gpu 同步 可刪除
        
        # for k in results.keys():
        #     dist.reduce(results[k], dst=0)
        # dist.reduce(nsamples, dst=0)
        
        # if rank == 0:
        #     logger.info('==========================================================================================')
        #     logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
        #     logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
        #     logger.info('==========================================================================================')
        #     print()
            
        #     for name, metric in results.items():
        #         writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        # for k in results.keys():
        #     if k in ['d1', 'd2', 'd3']:
        #         previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        #     else:
        #         previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        # if rank == 0:
        #     checkpoint = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'epoch': epoch,
        #         'previous_best': previous_best,
        #     }
        #     torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

        # === 輸出與記錄 ===
        if rank == 0:  # 主 GPU 或單 GPU
            logger.info("=" * 100)
            for h in head_names:
                logger.info(f"---- Head {h} ----")
                logger.info(
                    "{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(*tuple(results[h].keys()))
                )
                logger.info(
                    "{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(
                        *tuple([(v / nsamples[h]).item() for v in results[h].values()])
                    )
                )
                logger.info("=" * 100)

                # 寫入 TensorBoard
                for name, metric in results[h].items():
                    avg_value = (metric / nsamples[h]).item()
                    writer.add_scalar(f"eval/{h}_{name}", avg_value, epoch)

        # === 更新最佳結果 (previous_best)
        for h in head_names:
            for k in results[h].keys():
                avg_value = (results[h][k] / nsamples[h]).item()
                
                if k in ["d1", "d2", "d3"]:  # 越大越好
                    previous_best[h][k] = max(previous_best[h][k], avg_value)
                else:                        # 越小越好
                    previous_best[h][k] = min(previous_best[h][k], avg_value)

        # === 儲存 checkpoint ===
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": {
                    "l1": optim_l1.state_dict(),
                    "l3": optim_l3.state_dict(),
                    "l5": optim_l5.state_dict(),
                    "l7": optim_l7.state_dict(),
                    "backbone": optim_backbone.state_dict(),
                },
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, f"latest_epoch{epoch}.pth"))


if __name__ == '__main__':
    main()