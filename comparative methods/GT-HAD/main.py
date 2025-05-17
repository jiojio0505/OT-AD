# https://github.com/jeline0110/GT-HAD
import os
import numpy as np
import torch
import torch.optim as Optim
import scipy.io as sio
from net import Net
from sklearn.metrics import roc_auc_score, roc_curve
from utils import get_params, img2mask, seed_dict
import random
from progress.bar import Bar
import time
from thop import profile
import torch.nn as nn
from torch.utils.data import DataLoader
from data import DatasetHsi
from block import Block_fold, Block_search

dtype = torch.cuda.FloatTensor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir = '../data/'
save_dir = './results/'


def main(file):
    # set random seed
    # **************************************************************************************************************
    seed = seed_dict[file]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # data process
    # **************************************************************************************************************
    print(file)
    data_path = data_dir + file + '.mat'
    save_subdir = os.path.join(save_dir, file)
    if not os.path.exists(save_subdir):
        os.makedirs(save_subdir)

    # load data
    mat = sio.loadmat(data_path)
    img_np = mat['data']
    img_np = img_np.transpose(2, 0, 1)  # b, h, w
    img_np = img_np - np.min(img_np)
    img_np = img_np / (np.max(img_np) - np.min(img_np))  # [0, 1]
    gt = mat['map']
    img_var = torch.from_numpy(img_np).type(dtype)
    band, row, col = img_var.size()
    img_var = img_var[None, :]

    # set block functions and init dataloader
    # **************************************************************************************************************
    patch_size = 3
    patch_stride = 3
    block_size = patch_size * patch_stride  # block_size is the sliding window size
    data_set = DatasetHsi(img_var, wsize=block_size, wstride=3)
    block_fold = Block_fold(wsize=block_size, wstride=3)
    block_search = Block_search(img_var, wsize=block_size, wstride=3)
    data_loader = DataLoader(data_set, batch_size=64, shuffle=True, drop_last=False)
    data_num = data_set.__len__()

    # model setup
    # **************************************************************************************************************
    # net
    net = Net(in_chans=band, embed_dim=64, patch_size=patch_size,
              patch_stride=patch_stride, mlp_ratio=2.0, attn_drop=0.0, drop=0.0)
    net = net.cuda()

    # Calculate the parameters and FLOPs of the model (Optional)
    # dummy_block_idx = torch.ones(1, dtype=torch.int64).cuda()
    # dummy_match_vec = torch.zeros((data_num)).type(dtype).cuda()
    # dummy_input = torch.randn(1, band, block_size, block_size).cuda()
    # MACs, Params = profile(net, inputs=(dummy_input, dummy_block_idx, dummy_match_vec))
    # FLOPs = MACs * 2
    # print(f'Model FLOPs: {FLOPs / 1e9:.4f} GFLOPs')
    # print(f'Model Parameters: {Params / 1e6:.4f} M')

    # loss
    mse = torch.nn.MSELoss().type(dtype)

    # optim
    LR = 1e-3  # 2e-5
    p = get_params(net)
    optimizer = Optim.Adam(p, lr=LR)
    print('Starting optimization with ADAM')

    # train
    # **************************************************************************************************************
    end_iter = 150
    search_iter = 25  # [50, 100, 125]
    bar = Bar('Processing', max=end_iter)
    match_vec = torch.zeros((data_num)).type(dtype)
    search_matrix = torch.zeros((data_num, band, block_size, block_size)).type(dtype)
    search_index = torch.arange(0, data_num).type(torch.cuda.LongTensor)
    avgpool = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))  # k,s,p o=(i-k+2p)/s+1

    # start train
    start_time = time.time()
    for iter in range(1, end_iter + 1):
        search_flag = True if iter % search_iter == 0 and iter != end_iter else False
        for idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            # input -> net -> output
            net_gt, net_input, block_idx = batch_data['block_gt'], batch_data['block_input'], batch_data['index'].cuda()
            net_out = net(net_input, block_idx=block_idx, match_vec=match_vec)
            if search_flag: search_matrix[block_idx] = net_out
            # cal loss
            loss = mse(net_out, net_gt)
            loss.backward()
            optimizer.step()

        # CMM
        if search_flag:
            match_vec = torch.zeros((data_num)).type(dtype)  # reset match_vec
            search_back = block_fold(search_matrix.detach(), data_set.padding, row, col)
            match_vec = block_search(search_back.detach(), match_vec, search_index)
        bar.next()

        # start test
        if iter == end_iter:
            bar.finish()
            infer_loader = DataLoader(data_set, batch_size=64, shuffle=False, drop_last=False)
            net = net.eval()
            infer_res_list = []

            for idx, data in enumerate(infer_loader):
                infer_in = data['block_input']
                infer_idx = data['index'].cuda()
                # inference
                infer_out = net(infer_in, block_idx=infer_idx, match_vec=match_vec)
                infer_res = torch.abs(infer_in - infer_out) ** 2
                infer_res = avgpool(infer_res)
                infer_res_list.append(infer_res)

            infer_res_out = torch.cat(infer_res_list, dim=0)
            infer_res_back = block_fold(infer_res_out.detach(), data_set.padding, row, col)
            residual_np = img2mask(infer_res_back)

            # Calculate AUC
            auc_score = roc_auc_score(gt.flatten(), residual_np.flatten())
            print(f'AUC: {auc_score:.4f}')

            # Calculation time consumption
            total_time = time.time() - start_time
            print(f"Runtime: {total_time:.2f} seconds")
            print(f"-------------------------------------------------------------------------------------------")

            # Save the anomaly detection map and the ROC curve
            fpr, tpr, thresholds = roc_curve(gt.flatten(), residual_np.flatten())
            sio.savemat(os.path.join(save_dir, "Detection_map_GT-HAD.mat"), {'map': residual_np})
            sio.savemat(os.path.join(save_dir, "ROC_curve_GT-HAD.mat"), {'PD': tpr, 'FAR': fpr})

            return


if __name__ == "__main__":
    for file in ['abu-airport-4', 'abu-beach-3', 'abu-urban-2', 'abu-urban-3', 'abu-urban-4', 'hydice']:
        main(file)
