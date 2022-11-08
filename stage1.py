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
if __name__ == '__main__':
    # 构造多尺度类
    msi = MSI()
    # 加载 各种相互作用文件
    msi.load()
    print("load success")
    dp = DiffusionProfiles(
        alpha = 0.8595436247434408,
        max_iter = 1000,
        tol = 1e-06,
        weights = {
            'down_biological_function': 4.4863053901688685,
            'indication': 3.541889556309463,
            'biological_function': 6.583155399238509,
            'up_biological_function': 2.09685000906964,
            'protein': 4.396695660380823,
            'drug': 3.2071696595616364},
            num_cores = 6,
            save_load_file_path = "results/"
            )
    dp.calculate_diffusion_profiles(msi)