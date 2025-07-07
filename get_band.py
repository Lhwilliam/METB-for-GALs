#!/usr/bin/env python

# Toy graphene model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from __future__ import print_function
from pythtb import * # import TB model class
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ase.io import read
import pdb
k_open_C=np.zeros((1,2))
k_open_S=np.zeros((1,2))
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 18
for pp in range(1):     
    at = read('POSCAR')
    bond = np.loadtxt('bond.txt')
    C_num = len(np.where(at.get_atomic_numbers() == 6)[0])
    lat = at.cell[0:2,0:2]
    flag=0
    if at.cell[2,2]<0:
        tp= np.copy(lat[1,:]) 
        lat[1,:]=lat[0,:]
        lat[0,:]=tp
        flag=1 
     
    orb = at.positions[0:C_num,:] @ np.linalg.inv(at.cell)
    orb = orb[:,0:2]
    
    delta=0.0
    t=3 

    my_model_C = tb_model(2,2,lat,orb)
    my_model_S = tb_model(2,2,lat,orb)


    idx = 0
    all_dis = at.get_all_distances(mic=1)
    
    
    expand = at.positions[0:C_num,0:2]
    #import pdb;pdb.set_trace()
    for mm in range(-1,2):
        for nn in range(-1,2):
            if mm!= 0 or nn!= 0:
                expand = np.concatenate((expand, at.positions[0:C_num,0:2]+mm*at.cell[0,0:2]+nn*at.cell[1,0:2] ), axis=0) 
    #import p[MaEdb;pdb.set_trace()
    S_data = np.loadtxt('S_data.txt') 
    
    for ii in range(C_num):
        for jj in range(C_num*9):
            d = np.linalg.norm(at.positions[ii,0:2]-expand[jj,:])
            #import p[MaWdb;pdb.set_trace()
            if abs(d-1.4) < 0.3:
                idx += 1
                tp = np.array([ii, jj % C_num])
                neig_cell_idx = (expand[jj,:] - at.positions[ii,0:2]  + (at.positions[tp[0],0:2]-at.positions[tp[1],0:2])) @ np.linalg.inv(at.cell[0:2,0:2])
                neig_cell_idx = np.round(neig_cell_idx).astype(int)
                
                if tp[0] < tp[1]:
                    matches = np.all(S_data[:,0:2].astype(int)-1 == tp, axis = 1)
                    row_idx = np.where(matches)[0] 
                    #pdb.set_trace()
                    my_model_C.set_hop(t, tp[0],tp[1], [neig_cell_idx[0], neig_cell_idx[1]])
                    my_model_S.set_hop(-1.36858993-0.70311352*S_data[row_idx,3], tp[0],tp[1], [neig_cell_idx[0], neig_cell_idx[1]])
                   
    
    
    
    my_model_C.display()
    my_model_S.display()
    
    # generate list of k-points following a segmented path in the BZ
    # list of nodes (high-symmetry points) that will be connected

    path=np.array([[0,0.5],[0,0],[0.5,0.5 ],[2/3,1/3],[0.5,0],[0,0],[1/3,2/3]])
    if flag==1:
        path[:,[0,1]]=path[:,[1,0]]

    # labels of the nodes
    label=('M$_3$',r'$\Gamma $', 'M$_2$', 'K$_1$','M$_1$', r'$\Gamma $','K$_2$')
    # total number of interpolated k-points along the path
    nk=121
    
    # call function k_path to construct the actual path
    (k_vec,k_dist,k_node)=my_model_C.k_path(path,nk)
    
    print('---------------------------------------')
    print('starting C-calculation')
    print('---------------------------------------')
    print('Calculating bands...')
    
    # obtain eigenvalues to be plotted
    evals=my_model_C.solve_all(k_vec)
    dat = np.vstack((k_dist,evals))
    np.savetxt("tb-C-band-%d.dat"%pp,dat)
    


    v_band = dat[(dat.shape[0] // 2), :]  
    c_band = dat[(dat.shape[0] // 2 + 1), :]   
    Ec = np.min(c_band)  
    ia = np.argmin(c_band)  
    Ev = np.max(v_band)  
    ib = np.argmax(v_band)  


    if Ec < Ev:  
        gap = 0  
    else:  
        gap = Ec - Ev  
        if abs(ia - ib) < 5:  
            k_open_C[pp,:] = (k_vec[ia,:] + k_vec[ib,:]) / 2
            if flag ==1:
                   k_open_C[pp,[0,1]]=k_open_C[pp,[1,0]]
        else:  
            print('indirect')  
            k_open_C[pp,:] = [1000,1000]

    # figure for bandstucture
  
    fig, ax = plt.subplots()
    
    # specify horizontal axis details
    # set range of horizontal axis
    ax.set_xlim(k_node[0],k_node[-1])
    ax.set_ylim([-2,2])
    # put tickmarks and labels at node positions
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    # add vertical lines at node positions
    for n in range(len(k_node)):
      ax.axvline(x=k_node[n],linewidth=0.5, color='k')

    #ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Energy (eV)",fontname='Times New Roman')
    ax.spines['bottom'].set_linewidth(1.5)    
    ax.spines['left'].set_linewidth(1.5)     
    ax.spines['top'].set_linewidth(1.5)      
    ax.spines['right'].set_linewidth(1.5)   
    plt.gcf().set_size_inches(6,4)
    plt.tight_layout()
    # plot first and second band
    for i in range(C_num):
        if i == 0:
            #ax.plot(k_dist,evals[i],label='NNTB',linestyle='-',color='green',linewidth=1.5)
            ax.plot(k_dist,evals[i],label='NNTB',linestyle='-',color='#F3A332',linewidth=1.8)
        else:
            #ax.plot(k_dist,evals[i],linestyle='-',color='green',linewidth=1.2)
            ax.plot(k_dist,evals[i],linestyle='-',color='#F3A332',linewidth=1.8)
    fig.savefig("NNTB.png",dpi=450)

   


    # make an PDF figure of a plot
    #fig.tight_layout()
    # fig.savefig("GNM-compare-%d.png"%(pos_idx),dpi=450)
    print('C-Done.\n')

    
    print('---------------------------------------')
    print('starting S-calculation')
    print('---------------------------------------')
    print('Calculating bands...')
    
    # obtain eigenvalues to be plotted
    evals=my_model_S.solve_all(k_vec)
    dat = np.vstack((k_dist,evals))
    np.savetxt("tb-S-band-%d.dat"%pp,dat)
    

    v_band = dat[(dat.shape[0] // 2), :]
    c_band = dat[(dat.shape[0] // 2 + 1), :]
    Ec = np.min(c_band)
    ia = np.argmin(c_band)
    Ev = np.max(v_band)
    ib = np.argmax(v_band)

    if Ec < Ev:
        gap = 0
    else:
        gap = Ec - Ev
        if abs(ia - ib) < 5:
            k_open_S[pp,:] = (k_vec[ia,:] + k_vec[ib,:]) / 2
            if flag ==1:
                k_open_S[pp,[0,1]]=k_open_S[pp,[1,0]]
        else:
            print('indirect')
            k_open_S[pp,:] = [1000,1000]


    # figure for bandstucture
    
    fig0, ax0 = plt.subplots()
    # specify horizontal axis details

    # set range of horizontal axis
    ax0.set_xlim(k_node[0],k_node[-1])
    ax0.set_ylim([-2,2])
    # put tickmarks and labels at node positions
    ax0.set_xticks(k_node)
    ax0.set_xticklabels(label)
    # add vertical lines at node positions
    for n in range(len(k_node)):
        ax0.axvline(x=k_node[n],linewidth=0.5, color='k')
    ax0.set_xlabel("Path in k-space")
    ax0.set_ylabel("Energy (eV)",fontname='Times New Roman')

    
    # plot first and second band
    for i in range(C_num):
        if i == 0:
            #ax.plot(k_dist,evals[i],label='METB',linestyle='--',color='b',linewidth=1.8)
            ax.plot(k_dist,evals[i],label='METB',linestyle='-',color='#1868B2',linewidth=1.8)
            ax0.plot(k_dist,evals[i],label='METB',linestyle='--',color='b')
        else:
            #ax.plot(k_dist,evals[i],linestyle='--',color='b',linewidth=1.8)
            ax.plot(k_dist,evals[i],linestyle='-',color='#1868B2',linewidth=1.8)
            ax0.plot(k_dist,evals[i],linestyle='--',color='b')
    fig0.savefig("METB.png",dpi=600)
    
  
   

    DFT_dat = np.loadtxt('BAND.dat')
    k_points = DFT_dat[:, 0]  
    energies = DFT_dat[:, 1:] 
    n_bands = energies.shape[1]
    for i in range(n_bands):  
        if i == 0:
            #ax.plot(k_points/np.max(k_points)*k_node[-1], energies[:, i],label='DFT',linestyle='-.', color='r',linewidth=1.8)
            ax.plot(k_points/np.max(k_points)*k_node[-1], energies[:, i],label='DFT',linestyle='--', color='k',linewidth=1.8)  
        else:
            #ax.plot(k_points/np.max(k_points)*k_node[-1], energies[:, i],color='r',linewidth=1.8)    
            ax.plot(k_points/np.max(k_points)*k_node[-1], energies[:, i],color='#C72228',linewidth=1.8)
    legend=ax.legend(loc=6,facecolor=(1,1,1,1),framealpha=1,prop={'size': 14})

    # make an PDF figure of a plot
    #fig.tight_layout()
    
    fig.savefig("GNM-compare.png",dpi=600)
    print('S-Done.\n')

