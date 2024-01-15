#!/home/hvcl/anaconda3/envs/desc/bin/python

import sys
import json
import cgi


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

import csv
import time
import codecs




fs = cgi.FieldStorage()

	
sys.stdout.write("Content-Type: application/json")
sys.stdout.write("\n")
sys.stdout.write("\n")	
	
mfs = fs.getvalue("fd_ld")	
lines = mfs.split("_")




fd = int(lines[0])
ld = int(lines[1])
rds_name = lines[2]
rds_name_only = rds_name.replace('.rds', '')
nrow = int(lines[3])
n_dims = int(lines[4])
density_th = float(lines[5])

max_num_clusters = int(lines[6])


line_count = 7

ids = np.zeros(n_dims*max_num_clusters*nrow)

num_clusters_arr = np.zeros(n_dims)
num_points_arr = np.zeros(n_dims*max_num_clusters)

for i in range(n_dims):
    num_clusters = int(lines[line_count])
    num_clusters_arr[i] = num_clusters
    line_count += 1
    for j in range(num_clusters):
        num_points = int(lines[line_count])
        num_points_arr[i*max_num_clusters+j] = num_points
        line_count += 1
        for k in range(num_points):
            ids[i*max_num_clusters*nrow + j*nrow + k] = int(lines[line_count])
            line_count += 1







xy_size = nrow*n_dims




x = np.empty(xy_size)
y = np.empty(xy_size) 



dim_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/dimensions.csv'


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


density = np.zeros(n_dims*max_num_clusters*nrow)

x = x.astype(np.float32)
y = y.astype(np.float32)
ids = ids.astype(np.int32)
density = density.astype(np.float32)
num_clusters_arr = num_clusters_arr.astype(np.int32)
num_points_arr = num_points_arr.astype(np.int32)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)
ids_gpu = cuda.mem_alloc(ids.nbytes)
density_gpu = cuda.mem_alloc(density.nbytes)
num_clusters_arr_gpu = cuda.mem_alloc(num_clusters_arr.nbytes)
num_points_arr_gpu = cuda.mem_alloc(num_points_arr.nbytes)

cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)
cuda.memcpy_htod(ids_gpu, ids)
cuda.memcpy_htod(density_gpu, density)
cuda.memcpy_htod(num_clusters_arr_gpu, num_clusters_arr)
cuda.memcpy_htod(num_points_arr_gpu, num_points_arr)





mod = SourceModule("""

  
  __global__ void kde(float *x, float *y, int *ids, float *density, int *num_clusters_arr, int *num_points_arr, int nrow, int n_dims, int max_num_clusters)
  {
    
   
    
    
    double mpi = 3.141592654;
    
    
    int idx = (((gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;


    

    
    if (idx < n_dims*max_num_clusters*nrow)
    {
    
        int quotient = idx / nrow;
        int remainder = idx % nrow;
        
        int dim = quotient / max_num_clusters;
        int cluster = quotient % max_num_clusters;
        
        int num_clusters = num_clusters_arr[dim];
        
        if (cluster < num_clusters)
        {
            int num_points = num_points_arr[dim*max_num_clusters+cluster];
     


        
        
            if (remainder < num_points)
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
                
                
                
                for (int i=0; i<num_points; i++)
                {

                    double cur_x = x[dim*nrow + ids[dim*max_num_clusters*nrow + cluster*nrow + i]];
                    double cur_y = y[dim*nrow + ids[dim*max_num_clusters*nrow + cluster*nrow + i]];
                    
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
                    
                    
                    
                double bx = sum_x / num_points;
                double bx2 = sum_x2 / num_points;
                double sigma_x = sqrt(bx2 - bx*bx);
                double b_x = sigma_x * pow((3.0*num_points/4.0),(-1.0/5.0));
                bandwidth_x = b_x;
                
                double by = sum_y / num_points;
                double by2 = sum_y2 / num_points;
                double sigma_y = sqrt(by2 - by*by);
                double b_y = sigma_y * pow((3.0*num_points/4.0),(-1.0/5.0));
                bandwidth_y = b_y;
            

            
                double d = 0.0;
                
                double cur_x = x[dim*nrow + ids[dim*max_num_clusters*nrow + cluster*nrow + remainder]];
                double cur_y = y[dim*nrow + ids[dim*max_num_clusters*nrow + cluster*nrow + remainder]];
                
                for(int j = 0; j < num_points; j++)
                {
                    
                
                    double zx = (cur_x - x[dim*nrow + ids[dim*max_num_clusters*nrow + cluster*nrow + j]]) / bandwidth_x;
                    double ax = exp(-0.5*zx*zx) / (bandwidth_x * sqrt(2.0*mpi));
                    
        
        
                    double zy = (cur_y - y[dim*nrow + ids[dim*max_num_clusters*nrow + cluster*nrow + j]]) / bandwidth_y;
                    double ay = exp(-0.5*zy*zy) / (bandwidth_y * sqrt(2.0*mpi));
                    
                    d = d + ax*ay;
                }
                
        
        
                density[dim*max_num_clusters*nrow + cluster*nrow + remainder] = d / num_points;
            }
        }
    
 
    }
  }
""")



func = mod.get_function("kde")

func(x_gpu, y_gpu, ids_gpu, density_gpu, num_clusters_arr_gpu, num_points_arr_gpu, np.int32(nrow), np.int32(n_dims), np.int32(max_num_clusters), block=(32,32,1), grid=(1024,1024))





filtered_ids = np.full(n_dims*max_num_clusters*nrow, -1)

filtered_ids = filtered_ids.astype(np.int32)
filtered_ids_gpu = cuda.mem_alloc(filtered_ids.nbytes)
cuda.memcpy_htod(filtered_ids_gpu, filtered_ids)



mod2 = SourceModule("""

  
  __global__ void filter(int *ids, float *density, int *num_clusters_arr, int *num_points_arr, int *filtered_ids, int nrow, int n_dims, int max_num_clusters, float density_th)
  {
   
    
    
    int idx = (((gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (idx < n_dims*max_num_clusters*nrow)
    {
    


        int quotient = idx / nrow;
        int remainder = idx % nrow;
        
        int dim = quotient / max_num_clusters;
        int cluster = quotient % max_num_clusters;
        
        int num_clusters = num_clusters_arr[dim];
        

        if (cluster < num_clusters)
        {
            int num_points = num_points_arr[dim*max_num_clusters+cluster];
        
            if (remainder < num_points)
            {

                
                float max = INT_MIN;
                float min = INT_MAX;
                
                for (int i=0; i<num_points; i++)
                {
                    float cur_density = density[dim*max_num_clusters*nrow + cluster*nrow + i];
                    
                    if (cur_density > max)
                    {
                        max = cur_density;
                    }
                    
                    if (cur_density < min)
                    {
                        min = cur_density;
                    }
                }
            
                    
                float cur_thread_density = density[dim*max_num_clusters*nrow + cluster*nrow + remainder];
                
                
        
                if (((cur_thread_density - min) / (max - min)) > density_th)
                {    
                    filtered_ids[dim*max_num_clusters*nrow + cluster*nrow + remainder] = ids[dim*max_num_clusters*nrow + cluster*nrow + remainder];
                }
            } 
        }
    }
  }
""")



func2 = mod2.get_function("filter")
func2(ids_gpu, density_gpu, num_clusters_arr_gpu, num_points_arr_gpu, filtered_ids_gpu, np.int32(nrow), np.int32(n_dims), np.int32(max_num_clusters), np.float32(density_th), block=(32,32,1), grid=(1024,1024))



cuda.memcpy_dtoh(filtered_ids, filtered_ids_gpu)


filtered_ids_list = filtered_ids.tolist()



result = {}

ids_list = ids.tolist()
cuda.memcpy_dtoh(density, density_gpu)
density_list = density.tolist()


result['density'] = density_list




result['filtered_ids'] = filtered_ids_list
sys.stdout.write(json.dumps(result,indent=1))




sys.stdout.write("\n")

sys.stdout.close()



