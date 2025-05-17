# https://github.com/RSIDEA-WHU2020/Auto-AD
from __future__ import print_function
import os
import time
import torch.optim
import scipy.io as sio
from thop import profile
from sklearn.metrics import roc_auc_score, roc_curve
from utils.inpainting_utils import *
from models.skip import skip
from tools import hyper_norm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def main(kkkk):
    kkk = kkkk
    print(kkk)

    # Data input
    # **************************************************************************************************************
    torch.cuda.empty_cache()
    root_path = "../data/"
    residual_root_path = "./results/"
    file_name = root_path + str(kkk) + ".mat"
    save_dir = os.path.join(residual_root_path, kkk)
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    mat = sio.loadmat(file_name)
    hsi = mat['data']
    gt = mat['map']

    # Normalize
    img = hsi.transpose(2, 0, 1)  # (bands, height, width)
    img = hyper_norm(img)
    img = torch.from_numpy(img).type(dtype)
    bands, height, width = img.size()

    # Model setup
    # **************************************************************************************************************
    pad = 'reflection'  # 'zero'
    OPT_OVER = 'net'
    # OPTIMIZER = 'adam'
    method = '2D'
    input_depth = bands
    LR = 0.01
    num_iter = 1001
    param_noise = False
    reg_noise_std = 0.1  # 0 0.01 0.03 0.05
    thres = 0.000015
    channellss = 128
    layers = 5
    net = skip(input_depth, bands,
               num_channels_down=[channellss] * layers,
               num_channels_up=[channellss] * layers,
               num_channels_skip=[channellss] * layers,
               filter_size_up=3, filter_size_down=3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    net = net.type(dtype)  # see network structure
    net_input = get_noise(input_depth, method, img.size()[1:]).type(dtype)

    # Calculate the parameters and FLOPs of the model (Optional)
    # MACs, Params = profile(net, inputs=(net_input,))
    # FLOPs = MACs * 2
    # print(f'Model FLOPs: {FLOPs / 1e9:.4f} GFLOPs')
    # print(f'Model Parameters: {Params / 1e6:.4f} M')

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = img[None, :].cuda()

    mask_var = torch.ones(1, bands, height, width).cuda()
    residual_varr = torch.ones(height, width).cuda()

    def closure(iter_num, mask_varr, residual_varr):

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out_np = out.detach().cpu().squeeze().numpy()

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 100 == 0 and iter_num != 0:
            # Weighting block
            img_var_clone = img_var.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :])
            residual_img = temp.sum(0)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            # Residuals to weights
            residual_img = r_max - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):
                mask_var_clone[0, i, :] = residual_img[:]

        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone)
        total_loss.backward()
        print("iteration: %d; loss: %f" % (iter_num + 1, total_loss))

        return mask_var_clone, residual_var_clone, out_np, total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    end_iter = False
    p = get_params(OPT_OVER, net, net_input)
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)

    # Train
    start_time = time.time()
    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step()

        if j >= 1:
            index = j - int(j / 50) * 50
            loss_np[0][index - 1] = abs(loss - loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss

        if j == num_iter - 1 or end_iter is True:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()

            # Calculate AUC
            auc_score = roc_auc_score(gt.flatten(), residual_np.flatten())
            print(f'AUC: {auc_score:.4f}')

            # Calculation time consumption
            total_time = time.time() - start_time
            print(f"Runtime: {total_time:.2f} seconds")
            print(f"-------------------------------------------------------------------------------------------")

            # Save the anomaly detection map and the ROC curve
            fpr, tpr, thresholds = roc_curve(gt.flatten(), residual_np.flatten())
            sio.savemat(os.path.join(save_dir, "Detection_map_Auto-AD.mat"), {'map': residual_np})
            sio.savemat(os.path.join(save_dir, "ROC_curve_Auto-AD.mat"), {'PD': tpr, 'FAR': fpr})

            return


if __name__ == "__main__":
    for kkkk in ['abu-airport-4', 'abu-beach-3', 'abu-urban-2', 'abu-urban-3', 'abu-urban-4', 'hydice']:
        main(kkkk)

