import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
from random import sample


def get_discriminator(inp_dim, out_dim=1, hid_dim=64, n_hid_layers=2):
    s = nn.Sequential()
    s.add_module('i', nn.Linear(inp_dim, hid_dim))
    s.add_module('ai', nn.ReLU())
    for i in range(n_hid_layers):
        s.add_module(str(i), nn.Linear(hid_dim, hid_dim))
        s.add_module('a' + str(i), nn.ReLU())
    s.add_module('o', nn.Linear(hid_dim, out_dim))
    s.add_module('ao', nn.Sigmoid())
    return s


def d_loss_standard(batch_mix, batch_pos, discriminator, loss_function=None):
    d_mix = discriminator(batch_mix)
    d_pos = discriminator(batch_pos)
    if (loss_function is None) or (loss_function == 'log'):
        loss_function = lambda x: torch.log(x + 10 ** -10) # log loss
    elif loss_function == 'sigmoid':
        loss_function = lambda x: x  # sigmoid loss
    elif loss_function == 'brier':
        loss_function = lambda x: x ** 2 # brier loss
    return -(torch.mean(loss_function(1 - d_pos)) + torch.mean(loss_function(d_mix))) / 2


def d_loss_nnRE(batch_mix, batch_pos, discriminator, alpha, beta=0., gamma=1., loss_function=None):
    d_mix = discriminator(batch_mix)
    d_pos = discriminator(batch_pos)
    if (loss_function is None) or (loss_function == 'sigmoid'):
        loss_function = lambda x: 1 - x  # sigmoid loss
    elif loss_function == 'brier':
        loss_function = lambda x: (1 - x) ** 2 # brier loss
    pos_part = (1 - alpha) * torch.mean(loss_function(1 - d_pos))
    nn_part = torch.mean(loss_function(d_mix)) - (1 - alpha) * torch.mean(loss_function(d_pos))

    # return nn_part + pos_part, 1

    if nn_part.item() >= - beta:
        return pos_part + nn_part, 1
    else:
        return -nn_part, gamma


def train_NN(mix_data, pos_data, discriminator, d_optimizer, mix_data_test=None, pos_data_test=None,
             batch_size=64, n_epochs=200, n_batches=15, n_early_stop=5,
             d_scheduler=None, training_mode='standard', disp=False, loss_function=None, nnre_alpha=None):
    d_losses_train = []
    d_losses_test = []
    batch_size_mix = int(batch_size * mix_data.shape[0] / pos_data.shape[0])
    for epoch in range(n_epochs):
        d_losses_cur = []
        if d_scheduler is not None:
            d_sheduler.step()

        for i in range(n_batches):

            batch_mix = np.array(sample(list(mix_data), batch_size_mix))
            batch_pos = np.array(sample(list(pos_data), batch_size))

            batch_mix = torch.as_tensor(batch_mix, dtype=torch.float32)
            batch_pos = torch.as_tensor(batch_pos, dtype=torch.float32)
            batch_mix.requires_grad_(True)
            batch_pos.requires_grad_(True)

            # Optimize D
            d_optimizer.zero_grad()

            if training_mode == 'standard':
                loss = d_loss_standard(batch_mix, batch_pos, discriminator, loss_function=loss_function)
                loss.backward()
                d_optimizer.step()

            elif training_mode == 'nnre':
                loss, gamma = d_loss_nnRE(batch_mix, batch_pos, discriminator, nnre_alpha, beta=0.1, gamma=0.9, loss_function=loss_function)

                for param_group in d_optimizer.param_groups:
                    param_group['lr'] *= gamma

                loss.backward()
                d_optimizer.step()

                for param_group in d_optimizer.param_groups:
                    param_group['lr'] /= gamma
            d_losses_cur.append(loss.cpu().item())

        d_losses_train.append(round(np.mean(d_losses_cur), 5))

        if mix_data_test is not None and pos_data_test is not None:
            if training_mode == 'standard':
                d_losses_test.append(round(d_loss_standard(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                           torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                           discriminator).item(), 5))
            elif training_mode == 'nnre':
                d_losses_test.append(round(d_loss_nnRE(torch.as_tensor(mix_data_test, dtype=torch.float32),
                                                       torch.as_tensor(pos_data_test, dtype=torch.float32),
                                                       discriminator, nnre_alpha, beta=10)[0].item(), 5))

            if disp:
                print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1])

            if epoch > n_early_stop + 1:
                if_stop = True
                for i in range(n_early_stop):
                    if (d_losses_test[-(i + 1)] < d_losses_test[-(n_early_stop + 1)]):
                        if_stop = False
                        break
                if if_stop:
                    break
        elif disp:
            print('epoch', epoch, ', train_loss=', d_losses_train[-1])

    return d_losses_train, d_losses_test
