# https://github.com/DegangWang97/IEEE_TGRS_DirectNet
"""
See more details in papers:
  [1] D. Wang, L. Zhuang, L. Gao, X. Sun, X. Zhao, and A. Plaza, 
      “Sliding Dual-Window-Inspired Reconstruction Network for 
      Hyperspectral Anomaly Detection,” IEEE Trans. Geosci. Remote Sens., 
      vol. 62, 2024, Art. no. 5504115. DOI: 10.1109/TGRS.2024.3351179
      URL: https://ieeexplore.ieee.org/abstract/document/10400466

-----------------------------------------------------------------------
Copyright (January, 2024):  
            Degang Wang (wangdegang20@mails.ucas.ac.cn)
            Lina Zhuang (zhuangln@aircas.ac.cn)
            Lianru Gao (gaolr@aircas.ac.cn)
            Xu Sun (sunxu@aircas.ac.cn)
            Xiaobin Zhao (xiaobinzhao@bit.edu.cn)
            Antonio Plaza (aplaza@unex.es)

DirectNet is distributed under the terms of the GNU General Public License 2.0.

Permission to use, copy, modify, and distribute this software for
any purpose without fee is hereby granted, provided that this entire
notice is included in all copies of any software which is or includes
a copy or modification of this software and in all copies of the
supporting documentation for such software.
This software is being provided "as is", without any express or
implied warranty. In particular, the authors do not make any
representation or warranty of any kind concerning the merchantability
of this software or its fitness for any particular purpose.
---------------------------------------------------------------------
"""

import argparse
from model import DirectNet
from dataset import DirectNetData
from utils import get_auc, setup_seed, TensorToHSI, init_weights
import torch
from torch import optim
import torch.nn as nn
import scipy.io as sio
from sklearn.metrics import roc_curve
from torch.utils.tensorboard import SummaryWriter
import os
from thop import profile
import numpy as np
import time


class Trainer(object):
    '''
    Trains a model
    '''

    def __init__(self,
                 opt,
                 model,
                 criterion,
                 optimizer,
                 dataloader,
                 device,
                 model_path: str,
                 logs_path: str,
                 save_freq: int = 50,
                 scheduler=None):
        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training sets provided in `dataloader`
        using `criterion` and `optimizer`.
        Saves model weight snapshots every `save_freq` epochs and saves the
        weights at the end of training.
        Parameters
        ----------
        model : torch model object, with callable `forward` method.
        criterion : callable taking inputs and targets, returning loss.
        optimizer : torch.optim optimizer.
        dataloader : train dataloaders.
        model_path : string. output path for model.
        logs_path : string. output path for log.
        save_freq : integer. Number of epochs between model checkpoints. Default = 50.
        scheduler : learning rate scheduler.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.model_path = model_path
        self.logs_path = logs_path
        self.save_freq = save_freq
        self.scheduler = scheduler
        self.opt = opt

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.log_output = open(f"{self.logs_path}/log.txt", 'w')

        self.writer = SummaryWriter(logs_path)

        print(self.opt)
        print(self.opt, file=self.log_output)

    def train_epoch(self) -> None:
        # Run a train phase for each epoch
        self.model.train(True)
        loss_train = []

        for i, data in enumerate(self.dataloader):
            label = data['label'].to(self.device)
            input = data['input'].to(self.device)
            mask = data['mask'].to(self.device)

            # forward net
            output = self.model(input)

            # backward net
            self.optimizer.zero_grad()

            loss = self.criterion(output * (1 - mask), label * (1 - mask))

            loss.backward()
            self.optimizer.step()

            # get losses
            loss_train += [loss.item()]

            print("iter: " + str(i)
                  + "\tTrain Loss:" + str(round(np.mean(loss_train), 4)))

            print("iter: " + str(i)
                  + "\tTrain Loss:" + str(round(np.mean(loss_train), 4)), file=self.log_output)

        # ============ TensorBoard logging ============#
        # Log the scalar values
        info = {
            'Loss_train': np.mean(loss_train)
        }
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.epoch + 1)

        # Saving model
        if ((self.epoch + 1) % self.save_freq == 0):
            torch.save(self.model.state_dict(), os.path.join(self.model_path,
                                                             'DirectNet' + '_' + self.opt.dataset + '_' + str(
                                                                 self.epoch + 1) + '.pkl'))

    def train(self) -> nn.Module:
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs))
            print('Epoch {}/{}'.format(epoch + 1, self.opt.epochs), file=self.log_output)
            print('-' * 50)
            # run training epoch
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
        return self.model


def train_model(opt):
    DB = opt.dataset
    print(f'Processing: {DB}')

    expr_dir = os.path.join('./checkpoints/', DB)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    prefix = 'DirectNet' + '_batch_size_' + str(opt.batch_size) + '_epoch_' + str(opt.epochs) + '_learning_rate_' + str(
        opt.learning_rate) + \
             '_win_in_' + str(opt.win_in) + '_win_out_' + str(opt.win_out) + '_gpu_ids_' + str(opt.gpu_ids)

    trainfile = os.path.join(expr_dir, prefix)
    if not os.path.exists(trainfile):
        os.makedirs(trainfile)

    expr_dirs = os.path.join('./result/', DB)
    if not os.path.exists(expr_dirs):
        os.makedirs(expr_dirs)

    log_output = open(f"{expr_dirs}/log.txt", 'w')

    # Device
    device = torch.device('cuda:{}'.format(opt.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu')

    # Directories for storing model and loss
    model_path = os.path.join(trainfile, 'model')

    logs_path = os.path.join(trainfile, './logs')

    setup_seed(opt.seed)

    loader_train, band = DirectNetData(opt)
    net = DirectNet(band, band, nch_ker=opt.nch_ker, norm=opt.norm_mode, nblk=((opt.win_out - 7) // 4)).to(device)

    # Calculate the parameters and FLOPs of the model (Optional)
    # dummy_input = torch.randn(1, band, opt.win_out, opt.win_out).cuda()
    # MACs, Params = profile(net, inputs=(dummy_input,))
    # FLOPs = MACs * 2
    # print(f'Model FLOPs: {FLOPs / 1e9:.4f} GFLOPs')
    # print(f'Model Parameters: {Params / 1e6:.4f} M')

    # Initialize net parameters
    init_weights(net, init_type=opt.init_weight_type, init_gain=opt.init_gain)

    # Define Optimizers and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
    scheduler_net = None

    if opt.lossm.lower() == 'l1':
        criterion = nn.L1Loss().to(device)  # Regression loss: L1
    elif opt.lossm.lower() == 'l2':
        criterion = nn.MSELoss().to(device)  # Regression loss: L2

    if torch.cuda.is_available():
        print('Model moved to CUDA compute device.')
    else:
        print('No CUDA available, running on CPU!')

    # Training
    start_training_time = time.time()
    trainer = Trainer(opt,
                      net,
                      criterion,
                      optimizer,
                      loader_train,
                      device,
                      model_path,
                      logs_path,
                      scheduler=scheduler_net)
    trainer.train()
    total_training_time = time.time() - start_training_time
    print('Time of training-{}s'.format(total_training_time))
    print('Time of training-{}s'.format(total_training_time), file=log_output)


def predict(opt):
    DB = opt.dataset

    expr_dir = os.path.join('./checkpoints/', DB)
    prefix = 'DirectNet' + '_batch_size_' + str(opt.batch_size) + '_epoch_' + str(opt.epochs) + '_learning_rate_' + str(
        opt.learning_rate) + \
             '_win_in_' + str(opt.win_in) + '_win_out_' + str(opt.win_out) + '_gpu_ids_' + str(opt.gpu_ids)

    trainfile = os.path.join(expr_dir, prefix)

    model_path = os.path.join(trainfile, 'model')

    expr_dirs = os.path.join('./result/', DB)
    if not os.path.exists(expr_dirs):
        os.makedirs(expr_dirs)

    log_output = open(f"{expr_dirs}/log.txt", 'w')

    model_weights = os.path.join(model_path, 'DirectNet' + '_' + opt.dataset + '_' + str(opt.epochs) + '.pkl')

    # test dataloader
    data_dir = '../data/'
    image_file = data_dir + opt.dataset + '.mat'

    input_data = sio.loadmat(image_file)
    image = input_data['data']
    image = image.astype(np.float32)
    gt = input_data['map']
    gt = gt.astype(np.float32)

    band = image.shape[2]

    test_data = np.expand_dims(image, axis=0)
    loader_test = torch.from_numpy(test_data.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

    # Device
    device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

    net = DirectNet(band, band, nch_ker=opt.nch_ker, norm=opt.norm_mode, nblk=((opt.win_out - 7) // 4)).to(device)
    pretrained_dict = torch.load(model_weights)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    net.load_state_dict(pretrained_dict)

    start_eval_time = time.time()

    net.eval()
    img_old = loader_test
    test_data = loader_test.to(device)

    img_new = net(test_data)

    HSI_old = TensorToHSI(img_old)
    HSI_new = TensorToHSI(img_new)

    auc, detectmap = get_auc(HSI_old, HSI_new, gt)
    print("AUC: " + str(auc))
    print("AUC: " + str(auc), file=log_output)

    total_eval_time = time.time() - start_eval_time
    print('Time of testing-{}s'.format(total_eval_time))
    print('Time of testing-{}s'.format(total_eval_time), file=log_output)

    # Save the anomaly detection map and the ROC curve
    fpr, tpr, thresholds = roc_curve(gt.flatten(), detectmap.flatten())
    sio.savemat(os.path.join(expr_dirs, 'Detection_map_DirectNet.mat'), {'map': detectmap})
    sio.savemat(os.path.join(expr_dirs, "ROC_curve_DirectNet.mat"), {'PD': tpr, 'FAR': fpr})
    sio.savemat(os.path.join(expr_dirs, 'reconstructed_data.mat'), {'reconstructed_data': HSI_new})


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--gpu_ids", default=0, type=int, help='gpu ids: e.g. 0 1 2')

    parser.add_argument('--command', default='train', type=str, help='action to perform. {train, predict}.')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')

    parser.add_argument('--win_out', default=19, type=int, help='outer window size')
    parser.add_argument('--win_in', default=3, type=int, help='inner window size')

    parser.add_argument('--nch_ker', default=64, type=int, help='number of nch_ker')
    parser.add_argument('--norm_mode', choices=['bnorm', 'inorm'], default='bnorm', help='norm_mode to use')
    parser.add_argument('--init_weight_type', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], default='normal',
                        help='the name of an initialization method: normal | xavier | kaiming | orthogonal')
    parser.add_argument('--init_gain', default=0.02, type=float,
                        help='scaling factor for normal, xavier and orthogonal')

    parser.add_argument('--lossm', default='l1', type=str,
                        help='loss function for model training. one of ["l1", "l2"].')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='network parameter regularization')

    parser.add_argument('--epochs', default=700, type=int, help='number of epoch')
    parser.add_argument('--dataset', default='hydice', type=str,
                        help='dataset to use, abu-airport-4, abu-beach-3, abu-urban-2, abu-urban-3, abu-urban-4, hydice')

    opt = parser.parse_args()

    if opt.command.lower() == 'train':
        train_model(opt)
    elif opt.command.lower() == 'predict':
        predict(opt)
    return


if __name__ == '__main__':
    main()
