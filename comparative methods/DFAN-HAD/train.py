import torch
from torch import optim
import torch.utils.data as Data
import utils


def pre_DFAN(train_data, AE, args):
    epochs = args.pre_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    train_data = torch.from_numpy(train_data).float()
    train_load_data = Data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(AE.parameters(), lr=learning_rate)
    print("First Stage: Pretraining DFAN")
    for epoch in range(epochs):
        train_loss = 0
        AE.train()
        for idx, data in enumerate(train_load_data):
            data = data.cuda()
            z_c, output = AE(data, args)
            loss_function = torch.nn.MSELoss()
            loss = loss_function(output, data)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

    return z_c, AE,  args


def DFAN(data, model, args):
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    input = torch.from_numpy(data).float()
    train_data = Data.DataLoader(input, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Second Stage: Training DFAN")
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for idx, data in enumerate(train_data):
            data = data.cuda()
            lat_fea,  output = model(data, args)
            args.intra_dis, args.weight, args.inter_dis, intra_center_dis_mean = utils.AAM(lat_fea, args)
            loss_function = torch.nn.MSELoss(reduction='none')
            rec_loss_all_pixels = torch.mean(loss_function(output, data), dim=1)
            rec_loss = torch.matmul(args.weight.cuda(), rec_loss_all_pixels) / rec_loss_all_pixels.shape[0]
            intra_loss = args.intra_dis
            inter_loss = args.inter_dis
            intra_inter_loss_mean = intra_loss + inter_loss
            loss = rec_loss + intra_inter_loss_mean.cuda()
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

    return train_loss, model, args.weight


