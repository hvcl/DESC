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
dim_id = int(lines[5])

ids_string = lines[6].split(" ")

cell_num = int(ids_string[0])

ids = np.empty(cell_num, dtype=np.int32)



for i in range(cell_num):
    ids[i] = int(ids_string[i+1])









dim_path = 'data2/' + rds_name_only + '_' + str(fd) + '_' + str(ld) + '/dimensions.csv'


f = open(dim_path, 'r', encoding='utf-8')
rdr = csv.reader(f)

x_temp = np.empty(nrow, dtype=np.float32)
y_temp = np.empty(nrow, dtype=np.float32) 

r_count = 0

for line in rdr:
    if r_count!=0:
    
        x_temp[r_count-1] = float(line[dim_id*2+1]); 
        y_temp[r_count-1] = float(line[dim_id*2+2]);
    r_count = r_count + 1
	
f.close()   




x = np.empty(cell_num, dtype=np.float32)
y = np.empty(cell_num, dtype=np.float32)

for i in range(cell_num):
    x[i] = x_temp[ids[i]]
    y[i] = y_temp[ids[i]]
    

   
density = np.zeros(cell_num)


density = density.astype(np.float32)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)

density_gpu = cuda.mem_alloc(density.nbytes)

cuda.memcpy_htod(x_gpu, x)
cuda.memcpy_htod(y_gpu, y)

cuda.memcpy_htod(density_gpu, density)





mod = SourceModule("""

  
  __global__ void kde(float *x, float *y, float *density, int cell_num)
  {
    double mpi = 3.141592654;
    
    
    int idx = (((gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    

    if (idx < cell_num)
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
           
            double cur_x = x[i];
            double cur_y = y[i];
            
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
        
        double cur_x = x[idx];
        double cur_y = y[idx];
        
        for(int j = 0; j < cell_num; j++)
        {
            
            
            double zx = (cur_x - x[j]) / bandwidth_x;
            double ax = exp(-0.5*zx*zx) / (bandwidth_x * sqrt(2.0*mpi));
            
 
          
            double zy = (cur_y - y[j]) / bandwidth_y;
            double ay = exp(-0.5*zy*zy) / (bandwidth_y * sqrt(2.0*mpi));
            
            d = d + ax*ay;
        }
        

        
        density[idx] = d / cell_num;
    
    
    }
  }
""")



func = mod.get_function("kde")

func(x_gpu, y_gpu, density_gpu, np.int32(cell_num), block=(32,32,1), grid=(1024,1024))



cuda.memcpy_dtoh(density, density_gpu)


density_list = density.tolist()

result = {}


result['density'] = density_list
sys.stdout.write(json.dumps(result,indent=1))



sys.stdout.write("\n")

sys.stdout.close()



