# https://github.com/ChengXi-1217/DFAN-HAD
import os
import time
import test
import train
import torch
import model
import argparse
import random
import numpy as np
import scipy.io as sio
from thop import profile
from sklearn import preprocessing
from utils import seed_dic, hyper_norm
from sklearn.metrics import roc_auc_score, roc_curve


def HAD(args):
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
    mat = sio.loadmat(data_path)  # (height, width, bands)
    hsi = mat['data']
    GT = mat['map']

    # Normalize
    img_np = hyper_norm(hsi)
    img = torch.from_numpy(img_np).type(args.dtype)
    length, width, bands = img.size()
    img_2d = np.reshape(img_np, (length * width, bands))
    args.data = img_2d
    args.input_dim = bands
    args.length = length
    args.width = width
    args.GT = GT
    data = args.data  # 2D
    gt = args.GT
    latent_layer_dim = args.latent_layer_dim

    # Model
    DFAN = model.OrthoAE_Unet(args.input_dim, latent_layer_dim, args).cuda()

    # Calculate the parameters and FLOPs of the model (Optional)
    # dummy_args = args
    # dummy_input = torch.randn(length * width, bands).cuda()
    # MACs, Params = profile(DFAN, inputs=(dummy_input, dummy_args))
    # FLOPs = MACs * 2
    # print(f'Model FLOPs: {FLOPs / 1e9:.4f} GFLOPs')
    # print(f'Model Parameters: {Params / 1e6:.4f} M')

    # Train
    start_time = time.time()
    enc_fea, Pretrain_DFAN, args = train.pre_DFAN(data, DFAN, args)
    train_loss, Train_DFAN, weight = train.DFAN(data, Pretrain_DFAN, args)
    lat_fea, output = test.DFAN(data, Train_DFAN, args)

    rec_result = np.array(output.cpu(), dtype=float)
    rec_result = preprocessing.MinMaxScaler().fit_transform(rec_result)
    AD_result, _ = model.RX(rec_result - data)
    AD_result = AD_result.reshape((args.length, args.width))

    # Calculate AUC
    auc_score = roc_auc_score(gt.flatten(), AD_result.flatten())
    print(f'AUC: {auc_score:.4f}')

    # Calculation time consumption
    total_time = time.time() - start_time
    print(f"Runtime: {total_time:.2f} seconds")
    print(f"-------------------------------------------------------------------------------------------")

    # Save the anomaly detection map and the ROC curve
    fpr, tpr, thresholds = roc_curve(gt.flatten(), AD_result.flatten())
    sio.savemat(os.path.join(save_dir, "Detection_map_DFAN.mat"), {'map': AD_result})
    sio.savemat(os.path.join(save_dir, "ROC_curve_DFAN.mat"), {'PD': tpr, 'FAR': fpr})

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/', help='Data directory')
    parser.add_argument('--save_total_dir', type=str, default='./results/', help='Results save directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pre_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=40000)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--latent_layer_dim', type=int, default=64)
    parser.add_argument('--anomal_prop', type=float, default=0.003)
    parser.add_argument('--bandwidth', type=float, default=0.5)
    parser.add_argument('--dtype', type=str, default='cuda', help='Data type, cuda or cpu')
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--min_samples', type=int, default=3)
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
        HAD(args)
