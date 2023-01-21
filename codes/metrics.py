#!/home/hvcl/anaconda3/envs/desc/bin/python

import sys
import json
import cgi

from sklearn.metrics import (accuracy_score, adjusted_rand_score, calinski_harabasz_score, normalized_mutual_info_score, silhouette_score)
import numpy as np
import pandas as pd

import time

timestr = time.strftime("%Y%m%d-%H%M%S")

fs = cgi.FieldStorage()

sys.stdout.write("Content-Type: application/json")
sys.stdout.write("\n")
sys.stdout.write("\n")	

mfs = fs.getvalue("metrics_string")	
lines = mfs.split("_")


fd = int(lines[0])
ld = int(lines[1])

rds_name = lines[2]
rds_name_only = rds_name.replace('.rds', '')

cur_dim = int(lines[3])
timeString = lines[4]
participant = lines[5]


    
labels = lines[6].split(" ")


nrow = len(labels)
for i in range(nrow):
    labels[i] = int(labels[i])
 
labels = np.array(labels)    


gt = 0
gt_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/gt.csv'
if rds_name_only=="pbmc":
    gt = pd.read_csv(gt_path, header=None)
else:
    gt = pd.read_csv(gt_path)
gt = np.array(gt).reshape((gt.shape[0]))

pca_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/pca.csv'
pca = pd.read_csv(pca_path)
pca = np.array(pca.iloc[:, 0:cur_dim])


clu_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/clusterings.csv'
clusterings = pd.read_csv(clu_path)
clusterings = np.array(clusterings.iloc[:, (cur_dim-2)])



ARI_ours = round(adjusted_rand_score(gt, labels), 4)
NMI_ours = round(normalized_mutual_info_score(gt, labels), 4)
Silhouette_ours = silhouette_score(pca, labels)
Calinski_ours = calinski_harabasz_score(pca, labels)

ours = np.array([ARI_ours, NMI_ours, Silhouette_ours, Calinski_ours])
ours = pd.DataFrame(ours)
ours.index = ["ARI", "NMI", "Silhouette", "Calinski"]
ours_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/performance_ours_' + participant + '_' + timestr +  '.csv'
ours.to_csv(ours_path, index=True)



ARI_seurat = round(adjusted_rand_score(gt, clusterings), 4)
NMI_seurat = round(normalized_mutual_info_score(gt, clusterings), 4)
Silhouette_seurat = silhouette_score(pca, clusterings)
Calinski_seurat = calinski_harabasz_score(pca, clusterings)

seurat = np.array([ARI_seurat, NMI_seurat, Silhouette_seurat, Calinski_seurat])
seurat = pd.DataFrame(seurat)
seurat.index = ["ARI", "NMI", "Silhouette", "Calinski"]
seurat_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/performance_seurat_' + participant + '_' + timestr + '.csv'
seurat.to_csv(seurat_path, index=True)



time_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/start_time_' + participant + '_' + timeString + '.csv'
time_arr = np.array([])
time_arr = pd.DataFrame(time_arr)
time_arr.to_csv(time_path)


result = {}



sys.stdout.write(json.dumps(result,indent=1))


sys.stdout.write("\n")

sys.stdout.close()

