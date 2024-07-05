import numpy as np
import torch.optim
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import csv
import utils.loss
from utils.datasets import next_batch
from utils.evaluation import evaluation, get_cluster_sols
from models.baseModels import *
from utils.loss import *
from torch.nn.functional import normalize
from utils.util import target_l2
from utils.visualization import TSNE_show2D, TSNE_show3D, loss_plot
from models.MvFS_model import MvFS_DCN
from collections import Counter
# import sinkhornknopp as sk
# from models.SK import SinkhornKnopp
import time
from utils.graph_adjacency import get_similarity_matrix
from utils.map import xSimiarity
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import torch
import sinkhornknopp  as sk
class ICMCH(nn.Module):
    def __init__(self, config,para):
        super(ICMCH, self).__init__()
        self._config = config
        self._input_dim1 = config['Autoencoder']['gcnEncoder1'][0]
        self._input_dim2 = config['Autoencoder']['gcnEncoder2'][0]
        self._latent_dim = config['Autoencoder']['gcnEncoder1'][-1]
        # self._n_clusters = para["head"]
        self._n_clusters = config["n_clusters"]
        self.para=para
        self.gcnEncoder1 = GraphEncoder(config['Autoencoder']['gcnEncoder1'], 'relu', True)
        self.gcnEncoder2 = GraphEncoder(config['Autoencoder']['gcnEncoder2'], 'relu', True)

        self.instance_projector1 = InstanceProject(self._latent_dim)
        self.instance_projector2 = InstanceProject(self._latent_dim)
        self.class_loss=SupConLoss()
        self.cluster = ClusterProject(self._latent_dim, self._n_clusters)
        self.fusion = AttentionLayer(self._latent_dim)
        self.len=[128,128]
        self.feature_fusion = MvFS_DCN(self.len, self._latent_dim)
        self.sk=sk()
        # 标签优化
        N = len(self.train_loader.dataset)
        self.a = torch.full((N, 1), 1 / N).squeeze()
        self.args = para
        self.eta = self.args.eta
        self.b = torch.rand(self.args.classes, 1).squeeze()
        self.b = self.b / self.b.sum()
        self.ce_loss = nn.CrossEntropyLoss()
        self.u = None
        self.v = None
        self.h = torch.FloatTensor([1])
        # self.allb = [[self.b[i].item()] for i in range(self.args.classes)]
    def forward(self, x1, x2, adj1, adj2):
        h1 = self.gcnEncoder1(x1, adj1)
        h2 = self.gcnEncoder2(x2, adj2)
        h3 = self.gcnEncoder1(x1, adj1)
        h4 = self.gcnEncoder2(x2, adj2)
        z1 = normalize(self.instance_projector1(h1), dim=1)
        z2 = normalize(self.instance_projector2(h2), dim=1)
        # y1, p1 = self.cluster(h1)
        # y2, p2 = self.cluster(h2)
        y1, p1 = self.cluster(h3)
        y2, p2 = self.cluster(h4)
        return h1, h2, z1, z2, y1, y2, p1, p2

    def centorid(sellf, x,step,k):
        k = k # 假设我们想要的簇的数量
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x)

        # 获得簇心
        centroids = kmeans.cluster_centers_

        # 给每个簇心分配伪标签


        # 打印簇心和对应的伪标签
        return centroids
    def optimize_labels(self, step):
        # 1. aggregate P
        N = len(self.train_loader.dataset)
        PS = torch.zeros((N, self.args.classes))
        now = time.time()
        with torch.no_grad():
            for iter, (batch, selected) in enumerate(self.train_loader):
                input_ids, attention_mask = self.prepare_transformer_input(batch)
                emb1, _, _ = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)  # embedding
                p = F.softmax(self.model(emb1), dim=1)
                PS[selected] = p.detach().cpu()

        cost = -torch.log(PS)
        numItermax = 1000

        ###########
        if self.args.H == 'H2':
            #wang update b
            mu = 0.1
            z = torch.argmax(PS, dim=1)
            temp = list(range(self.args.num_classes))
            not_shown = list(set(temp).difference(set(z.numpy())))
            #print('not_shown ----:', len(not_shown))
            counts = Counter(z.numpy())
            for k in not_shown:
                counts[k] = 0
            self.b = mu * self.b + (1-mu) * torch.tensor(list(counts.values())) / sum(counts.values())
        #############
        T, log = sk.sinkhorn_knopp(self.a, self.b, cost, self.args.epsion, numItermax=numItermax, warn=False, log=True, u=self.u, v=self.v, h=self.h, reg2=self.args.reg2, log_alpha=self.args.logalpha, Hy=self.args.H)
        self.b = log['b']
        self.L = T
        print('Optimize Q takes {:.2f} min'.format((time.time() - now) / 60))

    def get_labels(self, step):
        # optimize labels
        print('[Step {}] Optimization starting'.format(step))
        # 更新self.L
        self.optimize_labels(step)
    def eval_acc(self, z, Y_list, accumulated_metrics, logger):
        """cal acc"""
        z = z.cpu().numpy()
        y_pred, _ = get_cluster_sols(z, ClusterClass=KMeans, n_clusters=self._n_clusters, init_args={'n_init': 10})
        scores = evaluation(y_pred=y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
        logger.info("\033[2;29m" + 'trainingset_view1 ' + str(scores) + "\033[0m")
    def soft_ce_loss(self, pred, target, step):
        # pred=torch.tensor(pred,dtype=torch.int64).cuda()
        tmp = target ** 2 / torch.sum(target, dim=0)
        target = tmp / torch.sum(tmp, dim=1, keepdim=True)
        if step >200:
            return torch.mean(-torch.sum(target * (F.log_softmax(pred, dim=1)), dim=1))
        else :
            return 0
    def run_train(self, x_train, Y_list, adj, optimizer, logger, accumulated_metrics, device):
        LOSS = []
        lamb1 = 1
        lamb2 = 1
        lamb3 = 1
        attention = True
        if lamb2 == 0:
            lamb3 = 0
            attention = False
        # epochs = self._config['training']['epoch']
        epochs=100
        print_num = self._config['print_num']
        batch_size = self._config['training']['batch_size']
        batch_size = batch_size if x_train[0].shape[0] > batch_size else x_train[0].shape[0]

        criterion_instance = InstanceLoss(batch_size, self.para["t1"], device).to(device)
        criterion_cluster = ClusterLoss(self._n_clusters, self.para["t2"], device).to(device)
        # criterion_instance=SupConLoss()
        # train the model

        for k in range(epochs):
            h1, h2, z1, z2, y1, y2, p1, p2 = self(x_train[0], x_train[1], adj[0], adj[1])

            if attention:

                h = self.fusion(h1, h2)

            else:
                h = 0.5 * (h1 + h2)

            # cluster contrastive loss
            x=ClusterProject(128,7)
            x=x(h)
            persudo_label=self.sk(x[0])
            persudo_label=torch.argmax(persudo_label,dim=1)
            y_3=  persudo_label
            if self.para["three"]=="all":
                cluster_loss = criterion_cluster(y1, y2)
                loss = lamb1 * cluster_loss

                # instance contrastive loss
                z1, z2,label = shuffle(z1, z2,Y_list[0])
                label=torch.from_numpy(label).to("cuda")
                for batch_z1, batch_z2,label1,batch_No in next_batch(z1, z2,label ,batch_size):

                    instance_loss = criterion_instance(batch_z1, batch_z2)


                    #
                    features = torch.cat([batch_z1.unsqueeze(1), batch_z2.unsqueeze(1)], dim=1)
                    # instance_loss=self.class_loss(features, label1)
                    loss += lamb2 * instance_loss

                    # loss += lamb2 * class_loss
                # high confidence loss
                y, _ = self.cluster(h)
                y = self.sk(y)
                y_1=self.sk(y_1)
                y_2=self.sk(y2)
                y_max = torch.maximum(y1, y2)
                y_max = torch.maximum(y_max, y)
                y_max = target_l2(y_max)
                # target=torch.argmax(y_max,dim=1).cuda()
                y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
                hc_loss = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
                # target = self.L[selected].cuda()
                hc_loss = self.soft_ce_loss(y_1, y, k) + self.soft_ce_loss( y_2,y1, k)+self.soft_ce_loss(y_3, y2, k)
                loss += lamb3 * hc_loss
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # LOSS.append(loss.item())


                # loss += lamb3 * hc_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                LOSS.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
            # csv_file = "tensor_data.csv"

            # 将张量按第一维度写入CSV文件

            if (k + 1) % print_num == 0:  # evaluation  k == 0 or
                output = ("Epoch:{:.0f}/{:.0f}===>loss={:.4f}".format((k + 1), epochs, loss.item()))
                logger.info("\033[2;29m" + output + "\033[0m")
                with torch.no_grad():
                    h1, h2, z1, z2, y1, y2, p1, p2 = self(x_train[0], x_train[1], adj[0], adj[1])
                    if attention:
                        h = self.fusion(h1, h2)
                        m=h.tolist()
                        csv_file = "Noisy"+str(k)+"tensor_data.csv"
                        label=Y_list[0].tolist()
                        for x in range(len(m)):
                            m[x].append(label[x])
                        with open(csv_file, mode='w', newline='') as file:
                            #
                            writer = csv.writer(file)
                            writer.writerows(m)
                    else:
                        h = 0.5 * (h1 + h2)
                    y, _ = self.cluster(h)

                    y = y.data.cpu().numpy().argmax(1)
                    scores = evaluation(y_pred=y, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
                    print(str(scores))
                    if lamb1 == 0:
                        self.eval_acc(h, Y_list, accumulated_metrics, logger)
        # loss_plot(LOSS, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ARI'])
        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1]

    def optimize_labels(self, step):
        # 1. aggregate P
        N = 210
        PS = torch.zeros((N, 7))


        with torch.no_grad():
            for iter, (batch, selected) in enumerate(self.train_loader):
                input_ids, attention_mask = self.prepare_transformer_input(batch)
                emb1, _, _ = self.model.get_embeddings(input_ids, attention_mask,
                                                       task_type=self.args.augtype)  # embedding
                p = F.softmax(self.model(emb1), dim=1)

                PS[selected] = p.detach().cpu()

        cost = -torch.log(PS)
        numItermax = 1000

        ###########
        if self.args.H == 'H2':
            # wang update b
            mu = 0.1
            z = torch.argmax(PS, dim=1)
            temp = list(range(self.args.num_classes))
            not_shown = list(set(temp).difference(set(z.numpy())))
            # print('not_shown ----:', len(not_shown))
            counts = Counter(z.numpy())
            for k in not_shown:
                counts[k] = 0
            self.b = mu * self.b + (1 - mu) * torch.tensor(list(counts.values())) / sum(counts.values())
        #############
        T, log = sk.sinkhorn_knopp(self.a, self.b, cost, self.args.epsion, numItermax=numItermax, warn=False, log=True,
                                   u=self.u, v=self.v, h=self.h, reg2=self.args.reg2, log_alpha=self.args.logalpha,
                                   Hy=self.args.H)
        self.b = log['b']
        self.L = T
        print('Optimize Q takes {:.2f} min'.format((time.time() - now) / 60))

    def get_labels(self, step):
        # optimize labels
        print('[Step {}] Optimization starting'.format(step))
        # 更新self.L
        self.optimize_labels(step)