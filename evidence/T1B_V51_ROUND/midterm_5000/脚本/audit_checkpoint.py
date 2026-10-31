import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,'D:/py/DRL2')
from policy_audit import audit
if __name__=='__main__':audit(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
