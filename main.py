import os
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import scipy.io as sio
from thop import profile
from progress.bar import Bar
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from dataset import Dataset, BlockRestore
from utils import seed_dic, hyper_norm
from net import Net


def run_single_file(args):
    # Set random seeds
    seed = seed_dic[args.file]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f'Processing: {args.file}')

    data_path = os.path.join(args.data_dir, f'{args.file}.mat')
    save_dir = os.path.join(args.save_total_dir, args.file)
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    mat = sio.loadmat(data_path)
    hsi = mat['data']
    gt = mat['map']

    # Normalize
    img = hsi.transpose(2, 0, 1)  # (bands, height, width)
    img = hyper_norm(img)
    img = torch.from_numpy(img).type(args.dtype).unsqueeze(0)
    bands, height, width = img.size()[1:]

    # DataLoader setup
    block_size = args.patch_size * args.patch_grid
    data_set = Dataset(img, block_size=block_size, stride=args.stride)
    block_restore = BlockRestore(block_size=block_size, stride=args.stride)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)

    # Model
    net = Net(
        input_channels=bands,
        embedding_dim=64,
        patch_size=args.patch_size,
        patch_grid=args.patch_grid,
        num_heads=args.num_heads,
        mlp_ratio=2.0,
        attn_drop_ratio=0.0,
        drop_ratio=0.0
    ).cuda()

    # Calculate the parameters and FLOPs of the model (Optional)
    # dummy_input = torch.randn(1, bands, args.patch_size * args.patch_grid, args.patch_size * args.patch_grid).cuda()
    # MACs, Params = profile(net, inputs=(dummy_input,))
    # FLOPs = MACs * 2
    # print(f'Model FLOPs: {FLOPs / 1e9:.4f} GFLOPs')
    # print(f'Model Parameters: {Params / 1e6:.4f} M')

    l1_loss = nn.L1Loss().type(args.dtype)
    optimizer = torch.optim.Adam(list(net.parameters()), lr=args.lr)

    # Train
    end_iter = args.num_iterations
    bar = Bar('Training', max=end_iter)
    start_time = time.time()
    for i in range(1, end_iter + 1):
        for data in train_loader:
            optimizer.zero_grad()
            net_in = data['block_input'].cuda()
            net_gt = data['block_gt'].cuda()
            net_out, w_loss = net(net_in)
            loss = l1_loss(net_out, net_gt) + w_loss
            loss.backward()
            optimizer.step()
        bar.next()

        # Test
        if i == end_iter:
            bar.finish()
            print(f"Finished training iteration: {i}")
            net.eval()
            test_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
            res_map = []

            with torch.no_grad():
                for data in test_loader:
                    test_in = data['block_input']
                    test_out = net(test_in)
                    res = torch.abs(test_in - test_out[0])
                    res_map.append(res)

            res_map = torch.cat(res_map, dim=0)
            res_map = block_restore(res_map, data_set.padding, height, width)
            res_map = res_map[0].sum(0)
            res_map = res_map.detach().cpu().numpy()
            res_map = hyper_norm(res_map)

            # Calculate AUC
            auc_score = roc_auc_score(gt.flatten(), res_map.flatten())
            print(f'AUC: {auc_score:.4f}')

            # Calculation time consumption
            total_time = time.time() - start_time
            print(f"Runtime: {total_time:.2f} seconds")
            print(f"-------------------------------------------------------------------------------------------")

            # Save the anomaly detection map and the ROC curve
            fpr, tpr, thresholds = roc_curve(gt.flatten(), res_map.flatten())
            sio.savemat(os.path.join(save_dir, "Detection_map_OT-AD.mat"), {'map': res_map})
            sio.savemat(os.path.join(save_dir, "ROC_curve_OT-AD.mat"), {'PD': tpr, 'FAR': fpr})

            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperspectral anomaly detection with multiple files')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--save_total_dir', type=str, default='./results/', help='Results save directory')
    parser.add_argument('--patch_size', type=int, default=3, help='Patch size')
    parser.add_argument('--patch_grid', type=int, default=5, help='Patch grid size')
    parser.add_argument('--stride', type=int, default=5, help='Sliding window stride')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_iterations', type=int, default=150, help='Training iterations')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of heads in transformer')
    parser.add_argument('--dtype', type=str, default='cuda', help='Data type, cuda or cpu')
    parser.add_argument('--files', type=str, default='abu-airport-4, abu-beach-3, abu-urban-2, abu-urban-3, abu-urban-4, hydice',
                        help='Comma-separated list of files to process')

    args = parser.parse_args()

    # Set dtype
    if args.dtype == 'cuda':
        args.dtype = torch.cuda.FloatTensor
    else:
        args.dtype = torch.FloatTensor

    file_list = [f.strip() for f in args.files.split(',')]
    for filename in file_list:
        args.file = filename
        run_single_file(args)
