from msi.msi import MSI
from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import numpy as np
import pickle
import csv
import networkx as nx
from scipy import spatial
from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles


file_path = "./data/10_top_msi/" # 读取原版数据
file_save = './rank_org.csv'

# file_path = "./results/" # 读取自己计算的数据
# file_save = './rank_self.csv'

def save_csv(filename, data):
    with open(filename ,'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for row in data:
            w.writerow(row)

msi = MSI()
msi.load()



dp_saved = DiffusionProfiles(alpha = None,
                            max_iter = None,
                            tol = None,
                            weights = None,
                            num_cores = None,
                            save_load_file_path = file_path)

msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

if __name__ == "__main__":
    df = []
    line = []
    file_col_names=['drugname', 'value']
    df.append(file_col_names)

    Hyperlipoproteinemia = dp_saved.drug_or_indication2diffusion_profile["C0020479"]
    for drug_it in msi.drugs_in_graph:
        drug_arr = dp_saved.drug_or_indication2diffusion_profile[drug_it]
        # 取所有药物文件npy 与 固定疾病 Hyperlipoproteinemia 计算
        drug_col = spatial.distance.correlation(drug_arr, Hyperlipoproteinemia)
        line = [drug_it, drug_col]
        df.append(line)
    
    save_csv(file_save, df)