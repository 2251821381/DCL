import time

import pandas as pd
import torch.optim
from utils.util import *
from utils.std_utils import *
from utils.datasets import *
import collections
import warnings
from models.ICMCH import ICMCH
from utils.graph_adjacency import *
from config import get_config

warnings.simplefilter("ignore")


# handwritten,MSRC_v1,LandUse-21,Scene-15,NoisyMNIST
def main(para):
    # prepare
    test_time = 1
    device = get_device()
        # torch.set_num_threads(16)

    for flag in [2]:

      # for dim in [512, 256, 128, 64]:
        config = get_config(flag=flag)
        config['print_num'] = 10
        # config['dim']=dim
        # logger
        logger, plt_name = get_logger(config)
        logger.info('Dataset:' + str(config['dataset']))

        # Load data
        X_list, Y_list = load_data(config, train_dir=True)
        x1_train_raw = X_list[0]
        x2_train_raw = X_list[1]




    # for k in range(para["k"]):
        config['topk'] = para["k"]
        print('K neighbors',  para["k"])
        config['missing_rate'] = para["ms"]
        print('missing_rate', para["ms"])

        # mask the data
        np.random.seed(2023)
        mask = get_mask(x1_train_raw.shape[0], config['missing_rate'])
        x1_miss = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_miss = x2_train_raw * mask[:, 1][:, np.newaxis]

        # data and mask to device
        x1_train = torch.from_numpy(x1_miss).float().to(device)
        x2_train = torch.from_numpy(x2_miss).float().to(device)
        mask = torch.from_numpy(mask).long().to(device)

        # get adjacency
        adj1, _, adj2, _ = get_miss_adjacency(x1_train, x2_train, mask, x1_miss.shape[0], topk=config['topk'])
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()

        fold_acc, fold_nmi, fold_ari = [], [], []

        for data_seed in range(1, test_time + 1):

                    setup_seed(data_seed)
                    accumulated_metrics = collections.defaultdict(list)  # Accumulated metrics
                    # Build model
                    model = ICMCH(config,para)
                    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
                    model.to(device)
                    # Training
                    acc, nmi, ari = model.run_train([x1_train, x2_train], Y_list, [adj1, adj2],
                                                    optimizer, logger, accumulated_metrics, device)



                    fold_acc.append(acc)
                    fold_nmi.append(nmi)
                    fold_ari.append(ari)

        logger.info('--------------------Training over--------------------')
        acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)



        logger.handlers.clear()

    print('acc:', acc, ',nmi:', nmi, ',ari:', ari)

    return acc, nmi, ari


if __name__ == '__main__':
    #消融实验参数
    acc, nmi, ari = main(para)






