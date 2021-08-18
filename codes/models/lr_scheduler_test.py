import math
from collections import Counter
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepGradient_Restart(_LRScheduler):
    def __init__(self, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class CosineAnnealingGradient_Restart():
    def __init__(self, T_period, restarts=None, weights=None, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        self.last_epoch = last_epoch
        self.lr = self.restart_weights[0]

    def get_weight(self, last_epoch):
        if last_epoch == 0:
            return self.lr
        elif last_epoch in self.restarts:
            self.last_restart = last_epoch
            self.T_max = self.T_period[self.restarts.index(last_epoch)]
            self.lr = self.restart_weights[self.restarts.index(last_epoch)]
            return self.lr
        self.lr *= (1 + math.cos(math.pi * (last_epoch - self.last_restart) / self.T_max)) /\
               (1 + math.cos(math.pi * ((last_epoch - self.last_restart) - 1) / self.T_max))
        return  self.lr


if __name__ == "__main__":

    ##############################
    # Cosine Annealing Restart
    ##############################

    scheduler = CosineAnnealingGradient_Restart(T_period=[250, 250, 250, 250], restarts=[0,250,500,750],
                                          weights=[1,1,1,1], last_epoch=0)

    ##############################
    # Draw figure
    ##############################
    N_iter = 1000
    lr_l = list(range(N_iter))
    for i in range(N_iter):
        lr_l[i] = scheduler.get_weight(i)

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Title', fontsize=16, color='k')
    plt.plot(list(range(N_iter)), lr_l, linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.show()
