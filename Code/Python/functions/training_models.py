import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import random
import tqdm
import time

import matplotlib.pyplot as plt

from torch import optim
from models import Net_Final, Net_Final_BIG, ImagesP2, ImagesP4




def epoch_train(model, train_tables, test_tables, optimizer, criterion1, criterion2, \
                e, im_size, simple, weighted, clipped, single_inp, expanded):
    sum_loss = 0
    acc = []
    idx_failures = []
    dist = 1 / (im_size - 1)

    # train data
    for k, TABLE in enumerate(train_tables):

        data = np.load('./minmax_data/data_vector_{}.npy'.format(TABLE), allow_pickle=True).tolist()
        data_y = np.load('./minmax_data/data_vector_y_{}.npy'.format(TABLE), allow_pickle=True).tolist()

        idx = list(data.keys())
        nv = len(data[idx[0]]) - 1

        for b in range(0, len(idx), mini_batch_size):
            # idx of clients to analyse
            t_idx = idx[b:b + mini_batch_size]
            if clipped:
                x = np.zeros((len(t_idx), 2, im_size, im_size))
            else:
                if expanded:
                    x = np.zeros((len(t_idx), 2*nv + 1, im_size, im_size))
                else:
                    x = np.zeros((len(t_idx), nv + 1, im_size, im_size))

            loc_y = np.zeros((len(t_idx), 2))
            x_aux = []

            for k, cl in enumerate(t_idx):
                loc = int(data_y[cl])
                loc_y[k] = [data[cl][loc + 1][2], data[cl][loc + 1][3]]

                x_aux.append(torch.tensor(np.asarray(data[cl][1:])).type(torch.FloatTensor))
                x[k][0][int(data[cl][0][0] // dist)][int(data[cl][0][1] // dist)] = 1
                x[k][0][int(data[cl][0][2] // dist)][int(data[cl][0][3] // dist)] = -1

                if clipped:
                    for i in range(1,31):
                        x[k][1][int(data[cl][i][2] // dist)][int(data[cl][i][3] // dist)] = 1
                else:

                    if expanded:
                        for i in range(1, 31):
                            x[k][2 * i - 1][int(data[cl][i][2] // dist)][int(data[cl][i][3] // dist)] = 1
                            x[k][2 * i][int(data[cl][i][4] // dist)][int(data[cl][i][5] // dist)] = 1
                    else:
                        for i in range(1, 31):
                            x[k][i][int(data[cl][i][2] // dist)][int(data[cl][i][3] // dist)] = 1

            y = np.hstack([data_y[i] for i in t_idx])

            train_x = torch.tensor(x).type(torch.FloatTensor)
            train_x_aux = torch.stack(x_aux).type(torch.FloatTensor)
            train_y = torch.tensor(y).type(torch.LongTensor)
            train_y_aux = torch.tensor(loc_y).type(torch.FloatTensor)

            #  set gradient to zero
            optimizer.zero_grad()

            #  compute output
            if single_inp:
                if simple:
                    output2 = model(train_x)
                    batch_loss = criterion2(output2, train_y)
                else:
                    output1, output2 = model(train_x)
                    batch_loss = 100*weighted * criterion1(output1, train_y_aux) + (1 - weighted) * criterion2(output2,
                                                                                                           train_y)
            else:
                if simple:
                    output2 = model(train_x, train_x_aux)
                    batch_loss = criterion2(output2, train_y)
                else:
                    output1, output2 = model(train_x, train_x_aux)
                    # take into account difference in order of magnitude
                    batch_loss = 100*weighted * criterion1(output1, train_y_aux) + (1 - weighted) * criterion2(output2,
                                                                                                           train_y)

            batch_loss.backward()
            optimizer.step()

            sum_loss = sum_loss + batch_loss.item()
            _, a = torch.max(output2, 1)
            acc.append(float((train_y == a).sum()) / len(train_y))

    test_loss = 0
    test_acc = []

    # model.eval()
    for k, TABLE in enumerate(test_tables):
        data = np.load('./minmax_data/data_vector_{}.npy'.format(TABLE), allow_pickle=True).tolist()
        data_y = np.load('./minmax_data/data_vector_y_{}.npy'.format(TABLE), allow_pickle=True).tolist()

        idx = list(data.keys())
        # random.shuffle(idx)

        for b in range(0, len(idx), mini_batch_size):

            t_idx = idx[b:b + mini_batch_size]

            if clipped:
                x = np.zeros((len(t_idx), 2, im_size, im_size))
            else:
                if expanded:
                    x = np.zeros((len(t_idx), 2*nv + 1, im_size, im_size))
                else:
                    x = np.zeros((len(t_idx), nv + 1, im_size, im_size))

            loc_y = np.zeros((len(t_idx), 2))
            x_aux = []

            for k, cl in enumerate(t_idx):
                loc = int(data_y[cl])
                loc_y[k] = [data[cl][loc + 1][2], data[cl][loc + 1][3]]

                x_aux.append(torch.tensor(np.asarray(data[cl][1:])).type(torch.FloatTensor))
                x[k][0][int(data[cl][0][0] // dist)][int(data[cl][0][1] // dist)] = 1
                x[k][0][int(data[cl][0][2] // dist)][int(data[cl][0][3] // dist)] = -1

                if clipped:
                    for i in range(1,31):
                        x[k][1][int(data[cl][i][2] // dist)][int(data[cl][i][3] // dist)] = 1
                else:

                    if expanded:
                        for i in range(1, 31):
                            x[k][2 * i - 1][int(data[cl][i][2] // dist)][int(data[cl][i][3] // dist)] = 1
                            x[k][2 * i][int(data[cl][i][4] // dist)][int(data[cl][i][5] // dist)] = 1
                    else:
                        for i in range(1, 31):
                            x[k][i][int(data[cl][i][2] // dist)][int(data[cl][i][3] // dist)] = 1

            y = np.hstack([data_y[i] for i in t_idx])

            test_x = torch.tensor(x).type(torch.FloatTensor)
            test_x_aux = torch.stack(x_aux).type(torch.FloatTensor)
            test_y = torch.tensor(y).type(torch.LongTensor)
            test_y_aux = torch.tensor(loc_y).type(torch.FloatTensor)

            if single_inp:
                if simple:
                    output2 = model(test_x)
                    batch_loss = criterion2(output2, test_y)
                else:
                    output1, output2 = model(test_x)
                    batch_loss = weighted * criterion1(output1, test_y_aux) + (1 - weighted) * criterion2(output2,
                                                                                                          test_y)
            else:
                if simple:
                    output2 = model(test_x, test_x_aux)
                    batch_loss = criterion2(output2, test_y)
                else:
                    output1, output2 = model(test_x, test_x_aux)
                    batch_loss = weighted * criterion1(output1, test_y_aux) + (1 - weighted) * criterion2(output2,
                                                                                                          test_y)

            test_loss += batch_loss.item()
            _, a = torch.max(output2, 1)

            test_acc.append(float((test_y == a).sum()) / len(test_y))
            idx_failures += [t_idx[i] for i in np.where(test_y != a)[0]]

    print('\rEpoch {}. Train Loss: {:.3f} Accuracy: {:.3f} Test Loss: {:.3f} Accuracy: {:.3f}'.format(e + 1, sum_loss,
                                                                                                      np.mean(acc),
                                                                                                      test_loss,
                                                                                                      np.mean(
                                                                                                          test_acc)),
          end="")
    return sum_loss, np.sum(acc) / len(acc), test_loss, np.sum(test_acc) / len(test_acc), idx_failures


def evaluating_model(train_tables, test_tables, model, im_size, n_epochs,
                     simple=False, weighted=0.5, clipped=False, single_inp=False, expanded=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion2 = nn.CrossEntropyLoss()
    criterion1 = nn.MSELoss()

    loss, acc, test_loss, test_acc, idx_f, times = [], [], [], [], [], []

    for epoch in range(n_epochs):
        current_t = time.time()
        train_l, accuracy, test_l, test_a, idx_failures = \
            epoch_train(model, train_tables, test_tables, optimizer, criterion1, criterion2, \
                        epoch, im_size, simple, weighted, clipped, single_inp, expanded)

        times.append(time.time() - current_t)
        loss.append(train_l)
        test_loss.append(test_l)
        acc.append(accuracy)
        test_acc.append(test_a)
        idx_f.append(idx_failures)

    print('\nAverage time per epoch {:.3f}s +- {:.3f}'.format(np.mean(times), 2 * np.std(times)))

    max_acc = np.max(test_acc)
    iter_max = np.where(test_acc == max_acc)

    print('Max accuracy of {:.3f} achieved at epoch {}'.format(max_acc, iter_max[0][0]))

    return loss, acc, test_loss, test_acc, idx_f, times



if __name__=='__main__':


    random.seed(4916)
    global MAX_V, mini_batch_size, n_epochs, lr, inp_size
    MAX_V = 30
    mini_batch_size = 50

    KERNEL_SIZE = 3
    tables = list(range(1, 98))
    random.shuffle(tables)

    lr = 0.0001
    inp_size = 4 + MAX_V * 6

    train_tables, test_tables, validation_tables = \
        tables[:int(len(tables) * 0.6)], tables[int(len(tables) * 0.6):int(len(tables) * 0.9)], tables[int(
            len(tables) * 0.9):]


    """
    
    
    # CASE 1. SINGLE INPUT SINGLE OUTPUT;
    print('CASE 1. Single Input Single Output MODEL 1\n')
    total_acc, total_loss = [], []
    im_size = 30
    n_epochs = 20

    main_acc, main_loss = [], []
    for trial in range(5):
        model1 = ImagesP2(31, MAX_V, im_size)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model1, im_size, n_epochs, simple=True, clipped=False, single_inp=True)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))
    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model1.parameters()), sum(
        p.numel() for p in model1.parameters() if p.requires_grad)))

    # CASE 2. SINGLE INPUT DOUBLE OUTPUT;
    print('\nCASE 2. Single Input Double Output MODEL 2. WEIGHT = 0.9\n')
    total_acc, total_loss = [], []
    im_size = 30
    n_epochs = 20

    main_acc, main_loss = [], []
    for trial in range(5):
        model2 = ImagesP4(31, MAX_V, im_size)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model2, im_size, n_epochs, weighted=0.99, clipped=False,
                             single_inp=True)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))

    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model2.parameters()), sum(
        p.numel() for p in model2.parameters() if p.requires_grad)))

    # CASE 3. SINGLE INPUT DOUBLE OUTPUT;
    print('\nCASE 3. Single Input Double Output MODEL 2. WEIGHT = 0.5\n')
    total_acc, total_loss = [], []
    im_size = 30
    n_epochs = 20

    main_acc, main_loss = [], []
    for trial in range(5):
        model2 = ImagesP4(31, MAX_V, im_size)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model2, im_size, n_epochs, weighted=0.5, clipped=False,
                             single_inp=True)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))
    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model2.parameters()), sum(
        p.numel() for p in model2.parameters() if p.requires_grad)))
    
    """
    random.seed(4914)
    tables = list(range(1, 90))
    random.shuffle(tables)

    lr = 0.0001
    inp_size = 4 + MAX_V * 6

    train_tables, test_tables, validation_tables = \
        tables[:int(len(tables) * 0.6)], tables[int(len(tables) * 0.6):int(len(tables) * 0.9)], tables[int(
            len(tables) * 0.9):]

    # CASE 4. DOUBLE INPUT DOUBLE OUTPUT; IMAGE_SIZE EXPLORATION
    print('\nCASE 4. Double Input Double Output MODEL 3. IMAGE SIZE = 30\n')
    total_acc, total_loss = [], []
    im_size = 30
    kernel_size=3
    n_epochs = 15

    main_acc, main_loss = [], []
    for trial in range(4):
        model3 = Net_Final(31, MAX_V, im_size, KERNEL_SIZE)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model3, im_size, n_epochs, weighted=0.9, clipped=False)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))
    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model3.parameters()), sum(
        p.numel() for p in model3.parameters() if p.requires_grad)))

    
    # CASE 5. DOUBLE INPUT DOUBLE OUTPUT; IMAGE_SIZE EXPLORATION
    print('\nCASE 5. Double Input Double Output MODEL 3. IMAGE SIZE = 50\n')
    total_acc, total_loss = [], []
    im_size = 50

    main_acc, main_loss = [], []
    for trial in range(4):
        model3 = Net_Final(31, MAX_V, im_size, KERNEL_SIZE)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model3, im_size, n_epochs, weighted=0.99, clipped=False)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))
    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model3.parameters()), sum(
        p.numel() for p in model3.parameters() if p.requires_grad)))


    # CASE 6. DOUBLE INPUT DOUBLE OUTPUT; CLIPPED INFO
    print('\nCASE 6. Double Input Double Output MODEL 4. IMAGE SIZE = 30. 2 channels\n')
    total_acc, total_loss = [], []
    im_size = 30
    n_epochs = 15

    main_acc, main_loss = [], []
    for trial in range(4):
        model4 = Net_Final(2, MAX_V, im_size, KERNEL_SIZE)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model4, im_size, n_epochs, weighted=0.99, clipped=True)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))
    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model4.parameters()), sum(
        p.numel() for p in model4.parameters() if p.requires_grad)))

    # CASE 7. DOUBLE INPUT DOUBLE OUTPUT; CLIPPED INFO
    print('\nCASE 7. Double Input Double Output MODEL 4. IMAGE SIZE = 50. 2 channels\n')
    total_acc, total_loss = [], []
    im_size = 50


    main_acc, main_loss = [], []
    for trial in range(4):
        model4 = Net_Final(2, MAX_V, im_size, KERNEL_SIZE)
        loss, acc, test_loss, test_acc, idx_f, times = \
            evaluating_model(train_tables, test_tables, model4, im_size, n_epochs, weighted=0.99, clipped=True)

        main_loss.append(min(test_loss))
        main_acc.append(max(test_acc))
    total_acc.append(main_acc)
    total_loss.append(main_loss)
    print('Average accuracy {:.3f} +- {:.3f}. Av loss {:.3f}\n -------------'.format( \
        np.mean(main_acc), np.std(main_acc), np.mean(main_loss)))
    print('Num parameters: {}\t Num Trainable parameters: {}'.format(sum(p.numel() for p in model4.parameters()), sum(
        p.numel() for p in model4.parameters() if p.requires_grad)))


