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
marker_num = int(lines[5])
density_th = float(lines[6])

ids = lines[7].split(" ")
for i in range(len(ids)):
    ids[i] = int(ids[i])

ids = np.array(ids)




xy_size = nrow*n_dims

plot_num = marker_num*n_dims



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


  
density = np.zeros(plot_num*(nrow))

x = x.astype(np.float32)
y = y.astype(np.float32)

ids = ids.astype(np.int32)
density = density.astype(np.float32)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)

ids_gpu = cuda.mem_alloc(ids.nbytes)
density_gpu = cuda.mem_alloc(density.nbytes)

cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)
cuda.memcpy_htod(ids_gpu, ids)
cuda.memcpy_htod(density_gpu, density)





mod = SourceModule("""

  
  __global__ void kde(float *x, float *y, int *ids, float *density, int nrow, int n_dims, int plot_num, int marker_num)
  {
    
   
    
    
    double mpi = 3.141592654;
    
    
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
                
             
                double zx = (cur_x - x[dim*nrow + ids[marker*(1+nrow) + 1 + j]]) / bandwidth_x;
                double ax = exp(-0.5*zx*zx) / (bandwidth_x * sqrt(2.0*mpi));
                
     
         
                double zy = (cur_y - y[dim*nrow + ids[marker*(1+nrow) + 1 + j]]) / bandwidth_y;
                double ay = exp(-0.5*zy*zy) / (bandwidth_y * sqrt(2.0*mpi));
                
                d = d + ax*ay;
            }
            
    
          
            density[marker*n_dims*nrow + dim*nrow + remainder] = d / cell_num;
        }
        

    }
  }
""")



func = mod.get_function("kde")

func(x_gpu, y_gpu, ids_gpu, density_gpu, np.int32(nrow), np.int32(n_dims), np.int32(plot_num), np.int32(marker_num), block=(32,32,1), grid=(1024,1024))




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


result['density'] = density_list




result['filtered_ids'] = filtered_ids_list
sys.stdout.write(json.dumps(result,indent=1))



sys.stdout.write("\n")

sys.stdout.close()




