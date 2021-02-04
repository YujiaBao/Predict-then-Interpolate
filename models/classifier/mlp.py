import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, depth=1):
        super(MLP, self).__init__()

        modules = [nn.Dropout(dropout)]
        last_dim = input_dim
        for i in range(depth-1):
            modules.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU()
            ])
            last_dim = hidden_dim
        modules.append(nn.Linear(last_dim, output_dim))
        self._main = nn.Sequential(*modules)

    def forward(self, x, y=None, return_pred=False, grad_penalty=False):
        '''
            @param x: batch_size * ebd_dim
            @param y: batch_size

            @return acc
            @return loss
        '''
        logit = self._main(x)

        if y is None:
            # return prediction directly
            return F.log_softmax(logit, dim=-1)

        loss = F.cross_entropy(logit, y)

        acc = self.compute_acc(logit, y)

        if return_pred:
            return torch.argmax(logit, dim=1), loss
        elif grad_penalty:
            # return irm grad penalty
            dummy = torch.tensor(1.).cuda().requires_grad_()
            loss_tmp = F.cross_entropy(logit * dummy, y)
            grad = torch.autograd.grad(loss_tmp, [dummy], create_graph=True)[0]
            return acc, loss, torch.sum(grad**2)
        else:
            return acc, loss

    @staticmethod
    def compute_acc(pred, true):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()
