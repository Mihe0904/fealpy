import numpy as np
import scipy.sparse as sp
np.set_printoptions(threshold=np.inf)

def ruge_stuben_chen_coarsen(A, theta=0.025):
    """
    @brief Long Chen 修改过的 Ruge-Stuben 粗化方法

    @param[in] A 对称正定矩阵
    @param[in] theta 粗化阈值
    """

    # 1. 初始化参数
    N = A.shape[0]
    isC = np.zeros(N, dtype=np.bool_)
    N0 = min(int(np.floor(np.sqrt(N))), 25)

    # 2. 生成强连通矩阵 
    # 然后函数计算出归一化的矩阵Am（矩阵A的对角线被归一化），
    # 并找出强连接的节点，也就是那些Am的元素值小于阈值theta的节点。
    # 得到的结果保存在矩阵G中。
    Dinv = sp.diags(1./np.sqrt(A.diagonal()))
    Am = Dinv @ A @ Dinv # 对角线归一化矩阵
    im, jm, sm = sp.find(Am)
    flag = (-sm > theta) 
    # 删除对角、非对角弱联接项，注意对角线元素为1，也会被过滤掉
    G = sp.csr_matrix((sm[flag], (im[flag], jm[flag])), shape=(N, N))

    # 3. 计算顶点的度 
    # 函数计算出每个节点的度，也就是与每个节点强连接的节点数量。
    # 如果有太多的节点没有连接，函数会随机选择N0个节点作为粗糙节点并返回。
    deg = np.array(np.sum(sp.csr_matrix(G, dtype=np.bool_), axis=1).flat,
            dtype=np.float64)
    if np.sum(deg > 0) < 0.25*np.sqrt(N):
        isC[np.random.choice(range(N), N0)] = True
        return isC, G

    flag = (deg > 0)
    deg[flag] += 0.1 * np.random.rand(np.sum(flag))

    # 4. 寻找最大独立集 
    # 函数尝试找出一个近似的最大独立集并将其节点添加到粗糙节点集合中。
    # 如果某节点被标记为粗糙节点，则其相邻的节点会被标记为细节点。
    isF = np.zeros(N, dtype=np.bool_)
    isF[deg == 0] = True # 孤立点为细节点 
    isU = np.ones(N, dtype=np.bool_) # 未决定的集合

    while np.sum(isC) < N/2 and np.sum(isU) > N0:
        # 如果粗节点的个数少于总节点个数的一半，并且未决定的点集大于 N0
        isS = np.zeros(N, dtype=np.bool_) # 选择集
        isS[deg>0] = True # 从非孤立点选择
        S = np.nonzero(isS)[0]
        # 非孤立点集的连接关系
        i, j = sp.triu(G[S, :][:, S], 1).nonzero()

        # 第 i 个非孤立点的度大于等于第 j 个非孤立点的度
        flag = deg[S[i]] >= deg[S[j]]
        isS[S[j[flag]]] = False # 把度小的节点从选择集移除
        isS[S[i[~flag]]] = False # 把度小的节点从选择集移除
        isC[isS] = True # 剩下的点就是粗点

        # Remove coarse nodes and neighboring nodes from undecided set
        i, _, _ = sp.find(G[:, isC])
        isF[i] = True # 粗点的相邻点是细点
        isU = ~(isF | isC) # 不是细点也不是粗点，就是未决定点
        deg[~isU] = 0 # 粗点或细节的度设置为 0

        if np.sum(isU) <= N0:
            # 如果未决定点的数量小于等于 N0，把未决定点设为粗点
            isC[isU] = True
            isU = []

    return isC, G

def ruge_stuben_coarsen(A, theta=0.025):
    """
    @brief Ruge-Stuben 粗化方法
    """
    N = A.shape[0]
    maxaij = A.min(axis=0)
    D = sp.diags(1/np.abs(maxaij).toarray().flatten())
    Am = D @ A

    # Delete weak connectness
    im, jm, sm = sp.find(Am)
    idx = (-sm > theta)
    As = sp.csr_matrix((np.ones_like(sm[idx]), (im[idx], jm[idx])), shape=(N, N))
    Am = sp.csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    Ass = (As + As.transpose()) / 2.0

    isF = np.zeros(N, dtype=bool)
    degIn = np.array(As.sum(axis=0)).flatten()
    #节点i的强连通点个数
    isF[degIn == 0] = True#判断孤立点

    # Find an approximate maximal independent set and put to C set
    isC = np.zeros(N, dtype=bool)
    U = np.arange(N)
    degFin = np.zeros(N)
    while np.sum(isC) < N / 2 and len(U) > 20:
        #强行终止条件：粗节点<20个或者粗节点大于等于一半细节点时
        isS = np.zeros(N, dtype=bool)
        degInAll = degIn + degFin
        #强连通点个数：在原本的基础上，叠加与经过处理点的联通数
        isS[(np.random.rand(N) < 0.85 * degInAll / np.mean(degInAll)) & (degInAll > 0)] = True
        S = np.where(isS)[0]
        #返回候选点的全局索引
        i, j, _ = sp.find(sp.triu(Ass[S][:, S], 1))
        #返回候选点的强连通情况，忽略对角线
        idx = degInAll[S[i]] >= degInAll[S[j]]
        #可能同时有两个相连的候选点，此时选择度数最大的那个候选点
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        #与当前粗节点强连通的点设为细节点
        i, _, _ = sp.find(Ass[:, isC])
        isF[i] = True
        U = np.where(~(isF | isC))[0]

        #将已标记为细节点或粗节点的度数设为0，避免重复处理。
        degIn[isF | isC] = 0
        degFin = np.zeros(N)
        #计算未决定节点与细节点之间的强连接数
        degFin[U] = np.array(As[isF, :][:, U].sum(axis=0)).flatten()

        if len(U) <= 20:
            isC[U] = True
            U = []

    print(f'Number of coarse nodes: {np.sum(isC)}')
    # return isC,Am


    allNode = np.arange(N)
    fineNode = allNode[~isC]
    Nf = len(fineNode)
    Nc = N - Nf

    coarseNode = np.arange(Nc)
    coarse2fine = np.where(isC)[0]
    fine2coarse = np.zeros(N, dtype=int)
    fine2coarse[isC] = coarseNode
    ip = coarse2fine
    jp = coarseNode
    sp_vals = np.ones(Nc)

    Afc = Am[fineNode, :][:, coarse2fine]
    Dsum = sp.diags(1 / np.array(Afc.sum(axis=1)).flatten())
    ti, tj, tw = sp.find(Dsum @ Afc)
    ip = np.concatenate((ip, fineNode[ti]))
    jp = np.concatenate((jp, tj))
    sp_vals = np.concatenate((sp_vals, tw))
    Pro = sp.coo_matrix((sp_vals, (ip, jp)), shape=(N, Nc))
    Res = Pro.transpose()

    # Ac = Res @ A @ Pro
    return Pro,Res

# def ruge_stuben_coarsen(A, theta=0.025):
#     """
#     Improved Python implementation of Ruge-Stuben coarsening 
#     with closer alignment to MATLAB's coarsenAMGrs
    
#     Parameters:
#         A : scipy.sparse.csr_matrix
#             Symmetric positive definite matrix
#         theta : float
#             Strong connection threshold
            
#     Returns:
#         Ac : scipy.sparse.csr_matrix
#             Coarse grid matrix
#         Pro : scipy.sparse.csr_matrix
#             Prolongation operator
#         Res : scipy.sparse.csr_matrix
#             Restriction operator
#     """
#     # ============== 1. Strong Connection Matrix ==============
#     N = A.shape[0]
    
#     # 1.1 Normalization by row maximum off-diagonal
#     maxaij = A.min(axis=0)
#     D = sp.diags(1/np.abs(maxaij).toarray().flatten())
#     # D = diags(1/np.abs(maxaij), 0)
#     Am = D @ A
    
#     # 1.2 Filter weak connections
#     im, jm, sm = sp.find(Am)
#     idx = (-sm > theta)
#     As = sp.csr_matrix((np.ones_like(sm[idx]), (im[idx], jm[idx])), shape=(N, N))
#     Ass = (As + As.T)/2  # Symmetrized version

#     # ============== 2. Coarse Node Selection ==============
#     isF = np.zeros(N, dtype=bool)
#     degIn = np.array(As.sum(axis=0)).flatten()
#     isF[degIn == 0] = True  # Isolated nodes -> fine
    
#     isC = np.zeros(N, dtype=bool)
#     U = np.where(~(isF | isC))[0]  # Undecided nodes
#     degFin = np.zeros(N)
    
#     while np.sum(isC) < N/2 and len(U) > 20:
#         # 2.1 Probabilistic selection
#         degInAll = degIn + degFin
#         prob = 0.85 * degInAll / np.mean(degInAll) if np.mean(degInAll) > 0 else 0
#         isS = (np.random.rand(N) < prob) & (degInAll > 0)
#         S = np.where(isS)[0]
        
#         # 2.2 Conflict resolution
#         rows, cols = sp.triu(Ass[S[:,None], S], 1).nonzero()
#         i, j = S[rows], S[cols]
#         mask = degInAll[i] >= degInAll[j]
#         isS[j[mask]] = False
#         isS[i[~mask]] = False
#         isC[isS] = True
        
#         # 2.3 Update F-set
#         neighbors = Ass[:, isC].nonzero()[0]
#         isF[neighbors] = True
#         U = np.where(~(isF | isC))[0]
        
#         # 2.4 Update degrees
#         degIn[isF | isC] = 0
#         degFin = np.zeros(N)
#         degFin[U] = np.array(As[isF,:][:,U].sum(axis=0)).flatten()
    
#     # Finalize remaining nodes
#     if len(U) <= 20:
#         isC[U] = True
    
#     # ============== 3. Prolongation Operator ==============
#     coarse_nodes = np.where(isC)[0]
#     fine_nodes = np.where(~isC)[0]
#     Nc = len(coarse_nodes)
    
#     # 3.1 Direct injection for C-points
#     ip = coarse_nodes
#     jp = np.arange(Nc)
#     sp_data = np.ones(Nc)
    
#     # 3.2 Interpolation weights for F-points
#     Afc = Am[fine_nodes,:][:,coarse_nodes]
#     row_sum = np.array(Afc.sum(axis=1)).flatten()
    
#     # Handle no-connection cases
#     zero_mask = (row_sum == 0)
#     if np.any(zero_mask):
#         # 仅修改受影响的行，而非整个矩阵
#         affected_rows = fine_nodes[zero_mask]
#         new_data = np.ones(len(affected_rows)*Nc)/Nc
#         new_rows = np.repeat(affected_rows, Nc)
#         new_cols = np.tile(jp, len(affected_rows))
        
#         # 合并原有数据和新数据
#         orig_rows, orig_cols = Afc.nonzero()
#         orig_data = Afc.data
        
#         ip = np.concatenate([ip, orig_rows, new_rows])
#         jp = np.concatenate([jp, orig_cols, new_cols])
#         sp_data = np.concatenate([sp_data, orig_data, new_data])
#     else:
#         # 正常情况下的处理
#         rows, cols, vals = sp.find(Afc)
#         row_sum = row_sum[rows]
#         ip = np.concatenate([ip, fine_nodes[rows]])
#         jp = np.concatenate([jp, cols])
#         sp_data = np.concatenate([sp_data, vals/row_sum])
    
#     # 3.3 Build operators
#     # print('ip',ip)
#     # print('jp',jp)
#     Pro = sp.csr_matrix((sp_data, (ip, jp)), shape=(N, Nc))
#     Res = Pro.T
#     Ac = Res @ A @ Pro
    
#     print(f'Number of coarse nodes: {np.sum(isC)}')
#     return Ac, Pro, Res

def aggregation_coarsen(A, theta=0.025):
    """
    @brief 
    """
    N = A.shape[0]
    isC = np.zeros(N, dtype=bool)
    N0 = min(int(np.sqrt(N)), 25)

    # Initialize output
    node2agg = np.zeros(N, dtype=int)
    agg2node = np.zeros(N, dtype=int)

    # Generate strong connectness matrix
    Dinv = sp.diags(1./np.sqrt(A.diagonal()))
    Am = Dinv @ A @ Dinv
    im, jm, sm = sp.find(Am)
    idx = (-sm > theta)
    As = sp.csr_matrix((sm[idx], (im[idx], jm[idx])), shape=(N, N))
    As += sp.eye(N)
    As1 = sp.csr_matrix(As, dtype=bool)
    As2 = sp.triu(As1 @ As1, 1).tocsr()

    # Compute degree of vertex
    deg = np.sum(As1, axis=1)
    deg = np.squeeze(np.asarray(deg))
    if np.sum(deg>0) < 0.25*np.sqrt(N):
        isC[np.random.choice(range(N), N0)] = True
        agg2node = np.where(isC)[0]
        node2agg[isC] = np.arange(len(agg2node))
        return node2agg, As

    idx = (deg>0)
    # 在计算 deg 时直接转成浮点数
    deg = np.sum(As1, axis=1).astype(float)  # 关键修改：转 float
    deg = np.squeeze(np.asarray(deg))

# 然后原来的加法就能正常运行
    deg[idx] += 0.1 * np.random.rand(np.sum(idx))

    # Find an approximate maximal independent set and put to C set
    isF = np.zeros(N, dtype=bool)
    isU = np.ones(N, dtype=bool)
    isS = np.ones(N, dtype=bool)
    isF[deg == 0] = True
    aggN = 0
    while aggN < N/2 and np.sum(isS) > N0:
        isS = np.zeros(N, dtype=bool)
        isS[deg>0] = True
        S = np.where(isS)[0]
        S_As2 = As2[S,:][:,S]
        i, j, _ = sp.find(S_As2)
        idx = deg[S[i]] >= deg[S[j]]
        isS[S[j[idx]]] = False
        isS[S[i[~idx]]] = False
        isC[isS] = True

        # Add new agg
        newC = np.where(isS)[0]
        newAgg = aggN + np.arange(len(newC))
        aggN += len(newC)
        node2agg[newC] = newAgg
        agg2node[newAgg] = newC

        # Remove coarse nodes and add neighboring nodes to the aggregate
        U = np.where(isU)[0]
        i, j, _ = sp.find(As[isU,:][:,newC])
        isF[U[i]] = True
        isU = ~(isF | isC)
        node2agg[U[i]] = node2agg[newC[j]]
        deg[newC] = 0
        deg[U[i]] = 0
        U = np.where(isU)[0]
        i, _, _ = sp.find(As[U,:][:,isF])
        deg[U[i]] = 0

    agg2node = agg2node[:max(node2agg)+1]

    # Add left vertices to existing agg
    while any(isU):
        U = np.where(isU)[0]
        i, j, _ = sp.find(As[:, isU])
        neighborAgg = node2agg[i]
        idx = (neighborAgg > 0)
        nAgg, neighborAgg = np.unique(neighborAgg[idx], return_counts=True)
        isbdU = (nAgg > 0)
        bdU = U[isbdU]
        node2agg[bdU] = neighborAgg[isbdU]
        isF[bdU] = True
        isU[bdU] = False

    return node2agg, As
