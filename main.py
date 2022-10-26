# 依赖引入
# 多尺度计算类
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

# 构造多尺度类
msi = MSI()
# 加载 各种相互作用文件
msi.load()
print("load success")
# print("run test msi")
# test_msi()

# 根据网络，对每个药物、疾病 扩散谱进行计算,作者已经计算好，无需进行这步
# 各参数含义详见论文
# Calculate diffusion profiles
# dp = DiffusionProfiles(
#         alpha = 0.8595436247434408, 
#         max_iter = 1, 
#         tol = 1e-06, 
#         weights = {
#                 'down_biological_function': 4.4863053901688685, 
#                 'indication': 3.541889556309463, 
#                 'biological_function': 6.583155399238509, 
#                 'up_biological_function': 2.09685000906964, 
#                 'protein': 4.396695660380823, 
#                 'drug': 3.2071696595616364
# 	}, 
# 	    num_cores = 24, 
#         save_load_file_path = "results/")
#         # 在 cpu <= 8 的电脑上，原（总数/2-4）计算方法CPU数量会计算出错，故使用总核心数/2
# 扩散谱计算入口
# dp.calculate_diffusion_profiles(msi)

# Load saved diffusion profiles
# 计算好的文件路径设置，直接加载扩散谱
dp_saved = DiffusionProfiles(
                alpha = None, 
                max_iter = None, 
                tol = None, 
                weights = None,
                num_cores = None, 
                save_load_file_path = "results/"
                )
# 加载保存的节点idx映射和节点列表 node2idx.pkl
msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
# 直接加载计算好的 疾病、药物的 扩散谱（会加载路径下所有.npy文件）
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

# Diffusion profile for Rosuvastatin (DB01098)
# 查看瑞舒伐他汀的扩散谱的概率数据
dp_saved.drug_or_indication2diffusion_profile["DB01098"]

# Test against reference
# 比较药物和疾病扩散谱的相似性来预测哪种药物对给定疾病有治疗效果
test_diffusion_profiles("data/10_top_msi/", "results/")