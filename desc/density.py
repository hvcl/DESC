#!/home/hvcl/anaconda3/envs/desc/bin/python

import sys
import json
import cgi
# import cudf
#import cgitb
# import os
# import time
# from cuml.cluster import DBSCAN
# import csv
# import numpy as np
# import math

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
import csv
import time
import codecs

# start = time.time()

# cell_num = 24911


# x = np.random.normal(size=1000)
# y = x * 3 + np.random.normal(size=1000)
# z = np.empty_like(x)



fs = cgi.FieldStorage()

	
sys.stdout.write("Content-Type: application/json")
sys.stdout.write("\n")
sys.stdout.write("\n")	
	
mfs = fs.getvalue("fd_ld")	
lines = mfs.split("_")
# array_len = len(lines)



fd = int(lines[0])
ld = int(lines[1])
rds_name = lines[2]
rds_name_only = rds_name.replace('.rds', '')
nrow = int(lines[3])
n_dims = int(lines[4])
marker_num = int(lines[5])
density_th = float(lines[6])

ids = lines[7].split(" ")
for i in range(len(ids)):
    ids[i] = int(ids[i])

ids = np.array(ids)



# nrow = 24911
# n_dims = 49
xy_size = nrow*n_dims
# marker_num = 1
plot_num = marker_num*n_dims



x = np.empty(xy_size)
y = np.empty(xy_size) 



dim_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/dimensions.csv'
# dim_path = 'dimensions.csv'

f = open(dim_path, 'r', encoding='utf-8')
rdr = csv.reader(f)

r_count = 0

for line in rdr:
    if r_count!=0:
        for i in range(n_dims):
            x[i*nrow + r_count-1] = float(line[i*2+1]); 
            y[i*nrow + r_count-1] = float(line[i*2+2]);
    r_count = r_count + 1
	
f.close()   


   
# cell_num = 10000

# id1 = np.random.randint(0, 24911, (cell_num))
# id1 = np.insert(id, 0, cell_num)

# id2 = np.random.randint(0, 24911, (cell_num))
# id2 = np.insert(id, 0, cell_num)

# ids = np.concatenate((id1, id2), axis=None)

# id_temp = np.random.randint(0, 24911, (cell_num))
# id1 = np.zeros(1 + nrow)

# id1[0] = cell_num
# for i in range(cell_num):
	# id1[i+1] = id_temp[i]
	
# id_temp = np.random.randint(0, 24911, (cell_num))
# id2 = np.zeros(1 + nrow)

# id2[0] = cell_num
# for i in range(cell_num):
	# id2[i+1] = id_temp[i]

# ids = np.concatenate((id1, id2), axis=None)

# print(ids.size)
# print(2+2*nrow)


# density = np.full((plot_num*(nrow)), -1)
density = np.zeros(plot_num*(nrow))

x = x.astype(np.float32)
y = y.astype(np.float32)
# z = z.astype(np.float32)
ids = ids.astype(np.int32)
density = density.astype(np.float32)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)
# z_gpu = cuda.mem_alloc(z.nbytes)
ids_gpu = cuda.mem_alloc(ids.nbytes)
density_gpu = cuda.mem_alloc(density.nbytes)

cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)
# cuda.memcpy_htod(z_gpu, z)
cuda.memcpy_htod(ids_gpu, ids)
cuda.memcpy_htod(density_gpu, density)


# print("time :", time.time() - start)
# start = time.time()


mod = SourceModule("""

  
  __global__ void kde(float *x, float *y, int *ids, float *density, int nrow, int n_dims, int plot_num, int marker_num)
  {
    
   
    
    
    double mpi = 3.141592654;
    
    
    int idx = (((gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // int xidx = (blockDim.x * blockIdx.x) + threadIdx.x;
    // int yidx = (blockDim.y * blockIdx.y) + threadIdx.y;
    // int idx = (gridDim.x * blockDim.x * yidx) + xidx;
    
    // int idx = 0;
    
    if (idx < plot_num*nrow)
    {
    
        int quotient = idx / nrow;
        int remainder = idx % nrow;
        
        int marker = quotient / n_dims;
        int dim = quotient % n_dims;
        
        
        // if (quotient < 2)    
        // {
     
        int cell_num = ids[marker*(1+nrow)];
     

        // int* dim_ids = new int[cell_num];
        
        // for (int i=0; i<cell_num; i++)
        // {
            // dim_ids[i] = ids[quotient*(1+nrow) + 1 + i];
            
        // }
        
        
        
   
        // double* dim_x = new double[cell_num];
        // double* dim_y = new double[cell_num];
        
        // for (int i=0; i<cell_num; i++)
        // {
            // dim_x[i] = x[quotient*nrow + dim_ids[i]];
            // dim_y[i] = y[quotient*nrow + dim_ids[i]];
        // }
        
        
        if (remainder < cell_num)
        {
            double sum_x = 0;
            double sum_x2 = 0;
            double min_map_x = 10000.0;
            double max_map_x = -10000.0;
            double bandwidth_x = -1.0;
            
            double sum_y = 0;
            double sum_y2 = 0;
            double min_map_y = 10000.0;
            double max_map_y = -10000.0;
            double bandwidth_y = -1.0;
            
            
            
            for (int i=0; i<cell_num; i++)
            {
                // double cur_x = dim_x[i];
                // double cur_y = dim_y[i];
                double cur_x = x[dim*nrow + ids[marker*(1+nrow) + 1 + i]];
                double cur_y = y[dim*nrow + ids[marker*(1+nrow) + 1 + i]];
                
                sum_x = sum_x + cur_x;
                sum_x2 = sum_x2 + cur_x*cur_x;
                
                if (cur_x < min_map_x)
                {
                    min_map_x = cur_x;
                }
                if (cur_x > max_map_x)
                {
                    max_map_x = cur_x;
                }
                
                sum_y = sum_y + cur_y;
                sum_y2 = sum_y2 + cur_y*cur_y;
                
                if (cur_y < min_map_y)
                {
                    min_map_y = cur_y;
                }
                if (cur_y > max_map_y)
                {
                    max_map_y = cur_y;
                }
            }	
                
                
                
            double bx = sum_x / cell_num;
            double bx2 = sum_x2 / cell_num;
            double sigma_x = sqrt(bx2 - bx*bx);
            double b_x = sigma_x * pow((3.0*cell_num/4.0),(-1.0/5.0));
            bandwidth_x = b_x;
            
            double by = sum_y / cell_num;
            double by2 = sum_y2 / cell_num;
            double sigma_y = sqrt(by2 - by*by);
            double b_y = sigma_y * pow((3.0*cell_num/4.0),(-1.0/5.0));
            bandwidth_y = b_y;
        

        
            double d = 0.0;
            
            double cur_x = x[dim*nrow + ids[marker*(1+nrow) + 1 + remainder]];
            double cur_y = y[dim*nrow + ids[marker*(1+nrow) + 1 + remainder]];
            
            for(int j = 0; j < cell_num; j++)
            {
                
                // double zx = (dim_x[remainder] - dim_x[j]) / bandwidth_x;
                double zx = (cur_x - x[dim*nrow + ids[marker*(1+nrow) + 1 + j]]) / bandwidth_x;
                double ax = exp(-0.5*zx*zx) / (bandwidth_x * sqrt(2.0*mpi));
                
     
                // double zy = (dim_y[remainder] - dim_y[j]) / bandwidth_y;
                double zy = (cur_y - y[dim*nrow + ids[marker*(1+nrow) + 1 + j]]) / bandwidth_y;
                double ay = exp(-0.5*zy*zy) / (bandwidth_y * sqrt(2.0*mpi));
                
                d = d + ax*ay;
            }
            
    
            // density[quotient*nrow + remainder] = d / cell_num;
            density[marker*n_dims*nrow + dim*nrow + remainder] = d / cell_num;
        }
        
        // delete [] dim_ids;
        // delete [] dim_x;
        // delete [] dim_y;
        // }
    }
  }
""")



func = mod.get_function("kde")
# func(x_gpu, y_gpu, z_gpu, np.int32(cell_num), block=(1,1,1))
# func(x_gpu, y_gpu, ids_gpu, density_gpu, np.int32(nrow), block=(2,1,1))
func(x_gpu, y_gpu, ids_gpu, density_gpu, np.int32(nrow), np.int32(n_dims), np.int32(plot_num), np.int32(marker_num), block=(32,32,1), grid=(1024,1024))

# print("time :", time.time() - start)
# start = time.time()



filtered_ids = np.full((plot_num*(nrow)), -1)

filtered_ids = filtered_ids.astype(np.int32)
filtered_ids_gpu = cuda.mem_alloc(filtered_ids.nbytes)
cuda.memcpy_htod(filtered_ids_gpu, filtered_ids)



mod2 = SourceModule("""

  
  __global__ void filter(int *ids, float *density, int *filtered_ids, int nrow, int n_dims, int plot_num, int marker_num, float density_th)
  {
   
    
    
    int idx = (((gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (idx < plot_num*nrow)
    {
    
        int quotient = idx / nrow;
        int remainder = idx % nrow;
        
        int marker = quotient / n_dims;
        int dim = quotient % n_dims;
        
        int cell_num = ids[marker*(1+nrow)];
        
        
        if (remainder < cell_num)
        {
            //float max = -1.0;
            //float min = 10000.0;
            
            float max = INT_MIN;
            float min = INT_MAX;
            
            for (int i=0; i<cell_num; i++)
            {
                float cur_density = density[marker*n_dims*nrow + dim*nrow + i];
                
                if (cur_density > max)
                {
                    max = cur_density;
                }
                
                if (cur_density < min)
                {
                    min = cur_density;
                }
            }
        
                   
            float cur_thread_density = density[marker*n_dims*nrow + dim*nrow + remainder];
            
            
            //if (((density[marker*n_dims*nrow + dim*nrow + remainder] / max)) > density_th)
            if (((cur_thread_density - min) / (max - min)) > density_th)
            {    
                filtered_ids[marker*n_dims*nrow + dim*nrow + remainder] = ids[marker*(1+nrow) + 1 + remainder];
            }
        } 
    }
  }
""")



func2 = mod2.get_function("filter")
func2(ids_gpu, density_gpu, filtered_ids_gpu, np.int32(nrow), np.int32(n_dims), np.int32(plot_num), np.int32(marker_num), np.float32(density_th), block=(32,32,1), grid=(1024,1024))



cuda.memcpy_dtoh(filtered_ids, filtered_ids_gpu)


filtered_ids_list = filtered_ids.tolist()



result = {}

ids_list = ids.tolist()
cuda.memcpy_dtoh(density, density_gpu)
density_list = density.tolist()

# result['ids'] = ids_list
result['density'] = density_list



# result['filtered_ids'] = filtered_ids
result['filtered_ids'] = filtered_ids_list
sys.stdout.write(json.dumps(result,indent=1))


# sys.stdout.write(json.dumps(filtered_ids_list, separators=(',', ':'), sort_keys=True, indent=4))

sys.stdout.write("\n")

sys.stdout.close()



# print(density[0]);

# print("time :", time.time() - start)



# dim_x1 = np.empty(cell_num)
# dim_y1 = np.empty(cell_num)
# dim_d1 = np.empty(cell_num)

# for i in range(cell_num):
	# dim_x1[i] = x[ids[1+i]]
	# dim_y1[i] = y[ids[1+i]]
	# dim_d1[i] = density[i]

# fig, ax = plt.subplots()
# ax.scatter(dim_x1, dim_y1, c=dim_d1, s=100, edgecolor='')
# plt.show()

# dim_x2 = np.empty(cell_num)
# dim_y2 = np.empty(cell_num)
# dim_d2 = np.empty(cell_num)

# for i in range(cell_num):
	# dim_x2[i] = x[nrow + ids[2+nrow+i]]
	# dim_y2[i] = y[nrow + ids[2+nrow+i]]
	# dim_d2[i] = density[nrow + i]

# fig, ax = plt.subplots()
# ax.scatter(dim_x2, dim_y2, c=dim_d2, s=100, edgecolor='')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=100, edgecolor='')
# plt.show()

