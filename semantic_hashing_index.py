# 2022.10.26

import faiss
import pickle
import numpy as np
import time


def make_binary_dataset(d, nt, nb, nq):
    '''
    生成测试数据
    d: dimension
    nt: test code num
    nb-nt: candidate code num
    nq-nb: query code num
    ===
    return:
    para1: the test code
    para2: the condidate code
    para3: the query code
    '''
    assert d % 8 == 0
    rs = np.random.RandomState(43)
    x = rs.randint(256, size=(nb + nq + nt, int(d / 8))).astype('uint8')
    return x[:nt], x[nt:-nq], x[-nq:]


def IndexBinaryFlat_for_random_data():
    '''
    a description of IndexBinaryFlat(int d, int b)
    使用前b个比特构建索引，哈希码的维度为d
    '''
    d = 16
    nq = 100
    nb = 2000
    (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)
    index_ref = faiss.IndexBinaryFlat(d)
    index_ref.add(xb)
    radius = 55
    # in Python, the results are returned as a triplet of 1D arrays lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
    Lref, Dref, Iref = index_ref.range_search(xq, radius)
    print("nb res: ", Lref[-1])


def bit2int(c):
    n = 0
    for x in c[:-1]:
        n = (n + x) << 1
    n = n + c[-1]
    return n


def bit2int8(codes):
    '''
    提取8个bit转为0-256之间的值
    '''
    bits_num = codes.shape[1]
    int8len = int(bits_num / 8)
    new_codes = []
    for code in codes:
        new_code = []
        for i in range(int8len):
            new_code.append(bit2int(code[i:(i+1)*8]))
        new_codes.append(new_code)
    new_codes = np.array(new_codes).astype('uint8')
    return new_codes


def IndexBinaryHash_for_ng20(dataset, bit_num):
    '''
    Using the VDSH model to generate hash codes of ng20 dataset 
    '''
    # load data
    with open('./metadata/train_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        train =pickle.load(file).cpu().numpy()
    with open('./metadata/train_label_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        train_label =pickle.load(file).cpu().numpy()
    with open('./metadata/test_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        test =pickle.load(file).cpu().numpy()
    with open('./metadata/test_label_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        test_label =pickle.load(file).cpu().numpy()
    print("code prepare...")
    train = bit2int8(train)
    test = bit2int8(test)
    print("code prepare ok!")
    index = faiss.IndexBinaryHash(bit_num, bit_num)
    index.add(train)
    # index.display()
    nfound = []
    ndis = []
    stats = faiss.cvar.indexBinaryHash_stats
    index.nflip = 2
    stats.reset()
    # D[i, j] contains the distance from the i-th query vector to its j-th nearest neighbor.
    # I[i, j] contains the id of the j-th nearest neighbor of the i-th query vector.
    D, I = index.search(test, 100)
    acc = 0
    for doc_ids, l in zip(I, test_label):
        for doc_id in doc_ids:
            if train_label[doc_id] == l:
                acc = acc + 1
    print("dataset={} b={}, p@100:".format(dataset, bit_num), acc / (test_label.shape[0]*100))


def IndexBinaryMultiHash_for_ng20(dataset, bit_num):
    '''
    using the vdsh output code
    '''
    # load data
    with open('./metadata/train_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        train =pickle.load(file).cpu().numpy()
    with open('./metadata/train_label_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        train_label =pickle.load(file).cpu().numpy()
    with open('./metadata/test_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        test =pickle.load(file).cpu().numpy()
    with open('./metadata/test_label_{}.tfidf_{}.pickle'.format(dataset, bit_num), 'rb') as file:
        test_label =pickle.load(file).cpu().numpy()
    print("code prepare...")
    train = bit2int8(train)
    test = bit2int8(test)
    print("code prepare ok!")
    if bit_num in [8, 16, 32, 64]:
        table_num = 4
    elif bit_num in [128,]:
        table_num = 8
    table_bit_num = int(bit_num / table_num)
    print(table_bit_num, bit_num // table_num)
    index = faiss.IndexBinaryMultiHash(bit_num, table_num, table_bit_num)
    print("index build ok!")
    index.add(train)
    print("data add ok!")
    index.display()
    stats = faiss.cvar.indexBinaryHash_stats
    index.nflip = table_bit_num
    print("begin to search!")
    D, I = index.search(test, 100)
    acc = 0
    for doc_ids, l in zip(I, test_label):
        for doc_id in doc_ids:
            if train_label[doc_id] == l:
                acc = acc + 1
    print("dataset={} b={}, p@100:".format(dataset, bit_num), acc / (test_label.shape[0]*100))

# index_hash_demo('ng20', 8)
# index_hash_demo('ng20', 16)
# index_hash_demo('ng20', 32)
    
# multi_index_hash_demo('ng20', 8)
# multi_index_hash_demo('ng20', 16)
# multi_index_hash_demo('ng20', 32)
# multi_index_hash_demo('ng20', 64)
multi_index_hash_demo('ng20', 128)

