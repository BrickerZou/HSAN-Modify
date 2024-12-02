from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import hard_sample_aware_network

if __name__ == '__main__':

    # for dataset_name in ["cora", "citeseer", "amap", "bat", "eat", "uat"]:
    for dataset_name in ["cora"]:

        # setup hyper-parameter
        args = setup_args(dataset_name)

        # record results
        file = open("result.csv", "a+")
        print(args.dataset, file=file)
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # ten runs with different random seeds
        for seed in range(5):
            # record results

            # fix the random seed
            setup_seed(seed)

            # load graph data 
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

            # apply the laplacian filtering  拉普拉斯滤波去噪
            X_filtered = laplacian_filtering(A, X, args.t)

            # test
            best_acc, best_nmi, best_ari, best_f1, y_hat, center = eva(X_filtered, y, cluster_num)

            # build our hard sample aware network
            HSAN = hard_sample_aware_network(
                input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num)

            # adam optimizer
            optimizer = optim.Adam(HSAN.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix 对角线为0，正样本的掩码
            mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

            # load data to device
            A, HSAN, X_filtered, mask = map(lambda x: x.to(args.device), (A, HSAN, X_filtered, mask))

            # training
            for epoch in tqdm(range(args.epochs), desc="training..."):
                # train mode
                HSAN.train()

                # encoding with Eq. (3)-(5) 生成两个特征矩阵嵌入，两个邻接矩阵嵌入
                Z1, Z2, E1, E2 = HSAN(X_filtered, A)

                # calculate comprehensive similarity by Eq. (6) 考虑结构和特征的综合相似度，S为两个图节点之间的相似度[node_num*2, node_num*2]
                S = comprehensive_similarity(Z1, Z2, E1, E2, HSAN.alpha)

                # calculate hard sample aware contrastive loss by Eq. (10)-(11) 动态权重，增大硬样本权重
                loss = hard_sample_aware_infoNCE(S, mask, HSAN.pos_neg_weight, HSAN.pos_weight, node_num)

                # optimization
                loss.backward()
                optimizer.step()

                # testing and update weights of sample pairs
                if epoch % 10 == 0:
                    # evaluation mode
                    HSAN.eval()

                    # encoding
                    Z1, Z2, E1, E2 = HSAN(X_filtered, A)

                    # calculate comprehensive similarity by Eq. (6)
                    S = comprehensive_similarity(Z1, Z2, E1, E2, HSAN.alpha)

                    # fusion and testing
                    Z = (Z1 + Z2) / 2
                    acc, nmi, ari, f1, P, center = eva(Z, y, cluster_num)
                    tqdm.write("Epoch: {:03d}, Loss: {:.4f}, ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}"
                               .format(epoch, loss.item(), acc, nmi, ari, f1))
                    # print(f"Z.shape: {Z.shape}")

                    # H高置信度样本, H_mat代表行和列标
                    H, H_mat = high_confidence_sample(Z, center)  # H.shape: torch.Size([2439])
                    # print(f"H.shape: {H.shape}, H_mat: {H_mat}")

                    # 动态权重机制，M记录正样本对权重，M_mat记录权重
                    M, M_mat = pseudo_matrix(P, S,
                                             node_num)  # M.shape: torch.Size([5416]) M_mat.shape: torch.Size([5416,5416])
                    # print(f"M.shape: {M.shape}, M_mat.shape: {M_mat.shape}")

                    # update weight
                    # 正样本对的权重（不同视图的相同节点）
                    HSAN.pos_weight[H] = M[H].data
                    # 正负样本对的权重（不同视图的不同节点）,相同节点的权重为0
                    HSAN.pos_neg_weight[H_mat] = M_mat[H_mat].data

                    # recording
                    if acc >= best_acc:
                        best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            file = open("result.csv", "a+")
            print("Seed {:02d}: Best ACC: {:.2f}, NMI: {:.2f}, ARI: {:.2f}, F1: {:.2f}"
                       .format(seed, best_acc, best_nmi, best_ari, best_f1))
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(best_acc, best_nmi, best_ari, best_f1), file=file)
            file.close()
            acc_list.append(best_acc)
            nmi_list.append(best_nmi)
            ari_list.append(best_ari)
            f1_list.append(best_f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        file = open("result.csv", "a+")
        print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
        file.close()
