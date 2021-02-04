from termcolor import colored
import torch
import numpy as np


def to_cuda(d):
    '''
        convert the input dict to cuda
    '''
    for k, v in d.items():
        d[k] = v.cuda()

    return d


def squeeze_batch(batch):
    '''
        squeeze the first dim in a batch
    '''
    res = {}
    for k, v in batch.items():
        assert len(v) == 1
        res[k] = v[0]

    return res


def print_res(train_res, val_res, ep):
    print(("epoch {epoch}, train {acc} {train_acc:>7.4f} "
           "{loss} {train_loss:>10.7f}, {regret} {train_regret:>10.7f} "
           "val {acc} {val_acc:>10.7f}, {loss} {val_loss:>10.7f}").format(
               epoch=ep,
               acc=colored("acc", "blue"),
               loss=colored("loss", "yellow"),
               regret=colored("regret", "red"),
               train_acc=train_res["acc"],
               train_loss=train_res["loss"],
               train_regret=train_res["regret"],
               val_acc=val_res["acc"],
               val_loss=val_res["loss"]), flush=True)


def get_worst_acc(true, pred, idx, loss, att_idx_dict):
    acc_list = (true == pred).float().tolist()
    idx = torch.cat(idx).tolist()
    idx_origin2new = dict(zip(idx, range(len(idx))))

    verbose = True
    if len(att_idx_dict) == 1:  # validation
        verbose = False

    worst_acc_list = []
    avg_acc_list = []

    for att, data_dict in att_idx_dict.items():
        if verbose:
            print('{:>20}'.format(att), end=' ')

        worst_acc = 1
        avg_acc = []
        for k, v in data_dict.items():
            # value to index mapping
            acc = []
            for origin in v:
                acc.append(acc_list[idx_origin2new[origin]])

            if len(acc) > 0:
                cur_acc = np.mean(acc)

                if verbose:
                    print(k, ' {:>8}'.format(len(v)),
                          ' {:>7.4f}'.format(cur_acc), end=', ')
                if cur_acc < worst_acc:
                    worst_acc = cur_acc
                avg_acc.append(cur_acc)

        if verbose:
            print(' worst: {:>7.4f}'.format(worst_acc))

        avg_acc = np.mean(avg_acc)
        worst_acc_list.append(worst_acc)
        avg_acc_list.append(avg_acc)

    # for worst, avg in zip(worst_acc_list, avg_acc_list):
    #     print('{:>7.4f}, {:>7.4f}'.format(worst, avg))

    return {
        'acc': np.mean(worst_acc_list),
        'avg_acc': np.mean(avg_acc_list),
        'loss': loss,
    }
