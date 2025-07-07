from ase.io import read
from sagar.io.vasp import read_vasp
from pyvaspflow.utils import get_perms
import numpy as np
import pdb

def get_bond(gy_idx):
    filename = 'POSCAR'
    at = read(filename)
    c = read_vasp(filename)
    
    
    perms = get_perms(c,symprec=0.4)
    
    C_num = len(np.where(at.get_atomic_numbers() == 6)[0])
    #C_num = len(at)

    dis_mat = at.get_all_distances(mic=1)
    bond = []
    
    # import pdb;pdb.set_trace()
    for ii in range(C_num):
        for jj in range(ii+1,C_num):
            if abs(dis_mat[ii,jj]-1.4) < 0.3:
                tmp = perms[:,[ii,jj]]
                tmp = tmp[np.lexsort(tmp[:,::-1].T),:][0]
                tmp = sorted(tmp)
                bond.append([ii+1,jj+1, tmp[0]+1, tmp[1]+1, dis_mat[ii,jj]])
    print(len(bond))
    np.savetxt('bond.txt',bond,fmt="%d %d %d %d %.4f")

if __name__ == "__main__":
    for i in range(1):
        get_bond(i) 
