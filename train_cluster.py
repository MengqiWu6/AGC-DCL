import os
import argparse

import torch
from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model,GraphConvolution
import torch.nn.functional as F
from sklearn.cluster import KMeans
from kmeans_gpu import pairwise_distance

from sklearn import manifold
from numpy import unique
from numpy import where
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
parser.add_argument('--dataset', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=6, help='type of dataset.')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--tau', type=float, default=1.0, help='temperature for Ncontrast loss')
parser.add_argument('--threshold', type=float, default=0.5, help='the threshold of high-confidence')   #0.5
parser.add_argument('--alpha', type=float, default=0.8, help='trade-off of loss')   #0.5  cora 0.8 uat 0.8 cite amap bat 1
#torch.device("cuda" if torch.cuda.is_available() else "cpu")
#parser.add_argument('--device', type=str, default='cpu', help='device')

args = parser.parse_args()

def get_feature_dis(x):
    #x :           batch_size x nhid
    #x_dis(i,j):   item means the similarity between x(i) and x(j).
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def Ncontrast(x_dis, adj_label, tau = 1):
    # compute the Ncontrast loss
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_A_r_flex(adj, r, cumulative=False):
    adj_d = adj.to_dense()
    adj_c = adj_d           # A1, A2, A3 .....
    adj_label = adj_d

    for i in range(r-1):
        adj_c = adj_c@adj_d
        adj_label = adj_label + adj_c if cumulative else adj_c
    return adj_label

#for args.dataset in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
for args.dataset in ["bat"]:
    print("Using {} dataset".format(args.dataset))
    file = open("result_baseline.csv", "a+")
    print(args.dataset, file=file)
    file.close()

    if args.dataset == 'cora':
        args.cluster_num = 7
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [500]
        args.beta = 2
    elif args.dataset == 'citeseer':
        args.cluster_num = 6
        args.gnnlayers = 2
        args.lr = 5e-5
        args.dims = [500]  #1000
        args.beta = 2
    elif args.dataset == 'amap':
        args.cluster_num = 8
        args.gnnlayers = 5
        args.lr = 1e-5
        args.dims = [500]
        args.beta = 3
    elif args.dataset == 'bat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [50]  #50
        args.beta = 5
    elif args.dataset == 'eat':
        args.cluster_num = 4
        args.gnnlayers = 5
        args.lr = 1e-3
        args.dims = [100] #100
        args.beta = 3
    elif args.dataset == 'uat':
        args.cluster_num = 4
        args.gnnlayers = 3
        args.lr = 1e-3
        args.dims = [100]  #100
        args.beta=6
    elif args.dataset == 'corafull':
        args.cluster_num = 70
        args.gnnlayers = 2
        args.lr = 1e-3
        args.dims = [500]

    # load data
    X, y, A = load_graph_data(args.dataset, show_details=False)
    features = X
    true_labels = y
    adj = sp.csr_matrix(A)

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()

    path = "../dataset/{}/{}_feat_sm_{}.npy".format(args.dataset, args.dataset, args.gnnlayers)
    if os.path.exists(path):
        sm_fea_s = sp.csr_matrix(np.load(path, allow_pickle=True)).toarray()
    else:
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)    #矩阵乘法
        np.save(path, sm_fea_s, allow_pickle=True)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []

    for seed in range(1):
        setup_seed(seed)
        best_acc, best_nmi, best_ari, best_f1, prediect_labels,dis,center = clustering(sm_fea_s, true_labels, args.cluster_num)
        dims=[features.shape[1]] + args.dims
        model = my_model(dims)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(args.device)


        inx = sm_fea_s.to(args.device)
        target = torch.FloatTensor(adj_1st).to(args.device)
        A_r=get_A_r_flex(target,r=2)

        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()

            z1, z2, fea_rec = model(inx, is_train=False, sigma=args.sigma)

            if epoch <210:


                S = z1 @ z2.T  # @表示矩阵乘法

                x_dis = get_feature_dis(S)
                nContrast_loss = Ncontrast(x_dis, A_r, tau=args.tau)

                loss1 = F.mse_loss(S, target)
                loss2 = F.mse_loss(fea_rec, inx)
                loss = loss1 + 20 * loss2    #20
                #loss=loss1
                loss.backward()
                optimizer.step()
                if epoch % 10 == 0:
                    model.eval()
                    z1, z2, fea_rec = model(inx, is_train=False, sigma=args.sigma)
                    hidden_emb = (z1 + z2) / 2

                    acc, nmi, ari, f1, predict_labels,dis,center = clustering(hidden_emb, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
            else:
                high_confidence = torch.min(dis, dim=1).values
                threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
                high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

                # pos samples
                index = torch.tensor(range(inx.shape[0]), device=args.device)[high_confidence_idx]
                y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]  # 高置信伪标签
                index = index[torch.argsort(y_sam)]  # torch.argsort 返回排序后的值对应原输入的下标
                class_num = {}

                for label in torch.sort(y_sam).values:
                    label = label.item()
                    if label in class_num.keys():
                        class_num[label] += 1
                    else:
                        class_num[label] = 1
                key = sorted(class_num.keys())
                if len(class_num) < 2:
                    continue
                pos_contrastive = 0
                centers_1 = torch.tensor([], device=args.device)
                centers_2 = torch.tensor([], device=args.device)

                for i in range(len(key[:-1])):     #遍历所有的标签类
                    class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
                    now = index[class_num[key[i]]:class_num[key[i + 1]]]
                    pos_embed_1 = z1[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]       #从now中选择80%的样本
                    pos_embed_2 = z2[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]


                    #S_pos=pos_embed_1@pos_embed_2.T     #维度为（n，n）对角线为i_v1和i_v2向量内积 （需要提取对角线）
                    #M = torch.abs(1 - S_pos) ** args.beta

                    S_pos=torch.sum(pos_embed_1 * pos_embed_2, dim=1)     #求i_v1和i_v2向量内积，维度为（n），
                    print('S_pos:',S_pos)
                    M = torch.abs(1 - S_pos) ** args.beta

                    pos_contrastive += (torch.exp(M)*(2 - 2 * S_pos)).sum()
                    pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()


                hidden_emb=(z1+z2)/2
                acc, nmi, ari, f1, predict_labels, dis,centers_1 = clustering(z1, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                acc, nmi, ari, f1, predict_labels, dis,centers_2 = clustering(z2, true_labels, args.cluster_num)
                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                '''
                if epoch % 10 == 0:
                    acc, nmi, ari, f1, predict_labels, dis, _ = clustering(hidden_emb, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
                '''
                pos_contrastive = pos_contrastive / args.cluster_num
                if pos_contrastive == 0:
                    continue
                if len(class_num) < 2:
                    loss = pos_contrastive
                else:
                    centers_1 = F.normalize(centers_1, dim=1, p=2)
                    centers_2 = F.normalize(centers_2, dim=1, p=2)
                    S = centers_1 @ centers_2.T  # @表示矩阵乘法
                    print('S_neg',S)

                    M_neg = torch.abs(0 - S) ** args.beta     #权重调制函数

                    S_diag = torch.diag_embed(torch.diag(S))  # torch.diag 取对角线元素 diag_embed指定元素变成对角矩阵
                    S = S - S_diag  # 去掉对角线元素

                    S=torch.exp(M_neg)*S    #e^M与S逐元素相乘

                    neg_contrastive = F.mse_loss(S, torch.zeros_like(S))
                    loss = pos_contrastive + args.alpha * neg_contrastive
                loss.backward()
                optimizer.step()
        high_similarity_indices = [0, 1, 4, 10, 15, 21, 24, 27, 30, 36, 50, 58, 71, 79, 84]
        print(torch.exp(M))
        print(torch.exp(M_neg))
        '''
        # hard_similarity
        similarity_matrix = torch.matmul(z1, z2.t())
        diagonal_similarities = similarity_matrix.diag()
        high_similarity_indices=[ 0,  1,  4, 10, 15, 21, 24, 27, 30, 36, 50, 58, 71, 79, 84]
        print(diagonal_similarities[high_similarity_indices])
        '''
        pos_embed_1 = z1[index]
        pos_embed_2 = z2[index]

        # 输出相似度高的节点索引
        similarity_matrix = torch.matmul(pos_embed_1, pos_embed_2.t())
        diagonal_similarities = similarity_matrix.diag()
        high_similarity_indices = torch.nonzero(diagonal_similarities < 0.8).squeeze()
        print(diagonal_similarities[high_similarity_indices])
        print(index[high_similarity_indices])


        '''
        high_similarity_indices=[52, 53, 56, 57, 60, 61]   #bat
        # high_similarity_indices=[337, 338, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 479, 480,
        # 481, 482, 483, 486, 489, 492, 496, 497, 500, 506, 508, 510, 512, 514,
        # 605, 611, 627, 662, 666, 669, 809, 810, 811, 812, 838, 839]
        high_similarity_indices=torch.tensor(high_similarity_indices)
        '''
        print(high_similarity_indices)

        plt.imshow(diagonal_matrix_np, cmap='Greys', interpolation='nearest', vmin=0.5, vmax=1)
        #plt.title('The visualization of the positive sample similarity')
        plt.colorbar()

        # 保存为图片
        plt.savefig('similarity_matrix_diagonal.png')
        plt.show()

        '''
                if epoch % 10 == 0:
                    model.eval()
                    z1, z2, fea_rec = model(inx, is_train=False, sigma=args.sigma)
                    hidden_emb = (z1 + z2) / 2
                    acc, nmi, ari, f1, predict_labels, dis, _ = clustering(z1, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
                    acc, nmi, ari, f1, predict_labels, dis, _ = clustering(z2, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
                    acc, nmi, ari, f1, predict_labels, dis, _ = clustering(hidden_emb, true_labels, args.cluster_num)
                    if acc >= best_acc:
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_f1 = f1
        '''
        '''
        ##visualization
        method = manifold.TSNE(n_components=2, init='pca', random_state=0)
        clusters = unique(predict_labels)
        hidden_emb=z1.detach().cpu().numpy()
        Y = method.fit_transform(hidden_emb)
        #ax = plt.figure(figsize=(8, 6)).add_subplot(211)    #211将画板一分为二（两行一列的子图网络），画在第一个子图上
        ax = plt.figure().add_subplot(111)
        for cluster in clusters:
            row_ix = where(predict_labels == cluster)
            ax.scatter(Y[row_ix, 0], Y[row_ix, 1], s=10)
        plt.savefig('amap_AGC-DCL.png', dpi=400)
        '''

        tqdm.write('acc: {}, nmi: {}, ari: {}, f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
        file = open("result_baseline.csv", "a+")
        print(best_acc, best_nmi, best_ari, best_f1, file=file)
        file.close()
        acc_list.append(best_acc)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)
        f1_list.append(best_f1)

    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)
    file = open("result_baseline.csv", "a+")
    print(args.gnnlayers, args.lr, args.dims, args.sigma, file=file)
    print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
    print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
    print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
    print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
    file.close()
