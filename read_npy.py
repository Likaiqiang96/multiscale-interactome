# 依赖引入
# 多尺度计算类
import csv
from msi.msi import MSI
# 扩散谱计算类
from diff_prof.diffusion_profiles import DiffusionProfiles
# 多线程 、 network 等计算库
import multiprocessing
import numpy as np
import pickle
import networkx as nx
# 测试函数
from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles

node_name = "DB01098"
file_path_org = "./data/10_top_msi/"
file_path_self = "./results/"

decode_org=r'./decode_org.csv'
decode_self=r'./decode_self.csv'

def save_csv(filename, data):
    with open(filename ,'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for row in data:
            w.writerow(row)

def save_npy_to_csv(msi, dp, node_name, filename):
    df = []
    line = []
    Go_n2i = dict()
    node_name_list = []
    file_col_names=['GO_index', 'node_index', 'value']
    
    for key_it in msi.node2idx:
        if 'GO' in key_it:
            Go_n2i[key_it] = msi.node2idx[key_it]
    npy_arr = dp.drug_or_indication2diffusion_profile[node_name]
    df.append(file_col_names)
    
    for go_it in Go_n2i:
        node_name_list.append(go_it)
    node_name_list.sort()

    for go_it in node_name_list:
        line=[go_it, Go_n2i[go_it], npy_arr[Go_n2i[go_it]]]
        df.append(line)
    save_csv(filename, df)

def read_npy_by_path(npy_path, node_name, result_file):
    dp_saved = DiffusionProfiles(
                    alpha = None, 
                    max_iter = None, 
                    tol = None, 
                    weights = None,
                    num_cores = None, 
                    save_load_file_path = npy_path
                    )
    # 加载保存的节点idx映射和节点列表 node2idx.pkl
    msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
    # 直接加载计算好的 疾病、药物的 扩散谱（会加载路径下所有.npy文件）
    dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

    # Diffusion profile for Rosuvastatin (DB01098)
    # 查看瑞舒伐他汀的扩散谱的概率数据
    # pass
    # arr = dp_saved.drug_or_indication2diffusion_profile["DB01098"]
    # print(list(arr)[0:100])
    save_npy_to_csv(msi, dp_saved, node_name, result_file)


if __name__ == '__main__':
    # 构造多尺度类
    msi = MSI()
    # 加载 各种相互作用文件
    msi.load()
    print("load success")
    read_npy_by_path(file_path_org, node_name, decode_org)
    read_npy_by_path(file_path_self, node_name, decode_self)
