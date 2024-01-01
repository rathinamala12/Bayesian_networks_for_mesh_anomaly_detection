from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.models import NoisyOrModel
from itertools import product
from itertools import tee                                                                                                             
import pandas as pd
import numpy as np
import time

# simulation for BLE mesh 12 node where each node speaks to client only through 3 hops


# simulation for BLE mesh 12 node where each node speaks to client  through relay nodes

#Source node is s1, vicinity range 0.6m


#s1--s4--s7--s10
#| \/| \/ | \/ |
#s2--s5--s8---c
#| \/| \/ | \/ |
#s3--s6--s9--s11


#Detangled mesh

#S1-->S4-->S7-->S10
#| /    \ |  \/  |
#V        V      V
#S2  S5  S8----->C
#  \    / |  \ / |
#         V      V
#S3  S6-->S9--->S11

U=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','C']

PU=['S1','S1','S3',['S1','S2'],'S5','S2','S4',['S4','S6','S7'],['S6','S8'],['S7','S8'],['S8','S9'],['S7','S8','S9','S10','S11']]  # parent nodes of  U
NP=[[1],[1],[0],[1,1],[0],[1],[1],[1,1,1],[1,1],[1,1],[1,1],[1,1,1,1,1]] # no of paths for node PU from source node S1
#NU=[1,1,1,2,2,3,1,3,3,1,3,9]
#print(len(U))
NU=np.zeros(len(U))
#NU[0]=1
pinhU=[]
for j in range(0,len(U)):
  
  for i in range(0,len(NP[j])):
      
      NU[j]=NU[j]+NP[j][i]
  p=int(NU[j]+1)
  pinhU.append([0]*p)
  #print(pinhU)   
  #print(NU[j])
  
u0=np.random.rand()
u1=np.random.rand()
u2=np.random.rand()
u3=np.random.rand()
u4=np.random.rand()
u5=np.random.rand()
u6=np.random.rand()

#print(u0)  
'''
  pinhU=[[0]*int(NU[j])]*len(U)
  print(pinhU)
'''
inh_prob=[u0,u0,[u0,u1],u3,u2,u4,[u4*u6*u7],[u1*u2],[u1*u3],[u1*u2],[u1*u2*u2*u3*u3]] #failure probabilities chosen as per orientation of source and destination nodes.
#print(inh_prob)

#inh_prob=[u0,0.3,0.2,0.2,0,0,0,0]
for j in range(0,len(U)):
    
  
    for i in range(0,int(NU[j])+1):
      if (i==0):
          pinhU[j][i]=inh_prob[j]
      else:
          pinhU[j][i]=0

#print(pinhU)
#print(pinhU[5][0])

#pinhU=[[0, 0], [0.3, 0], [0.2, 0], [0.2, 0], [0.06, 0, 0], [0.4, 0], [0.024, 0, 0, 0, 0]]

'''
E_0=[1,0.3,0.2,0.06]
E_1=[0,0.7,0.8,0.38]
E_2=[0,0,0,0.56]

S5_0=[1,0.2,0.2,0.2,0.3,0.1,0,0,0,0,0,0,0,0,0,0]
S5_1=[0,0.2,0.8,0.8,0.7,0.9,1,1,1,1,1,1,1,1,1,0]
S5_2=[0,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
S5_3=[0,0.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
S5_4=[0,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
S5_5=[0,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
S5_6=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
'''
start_random=time.time()
'''
v=np.arange(0,7)
S5=np.random.rand(7,16)
r=S5.sum(axis=0)
S5=S5/r
l=np.arange(0,17)
#print(l)
'''
#states of S8 depends on the state of the parents  S4,S6,S7
# S4 & S7 takes two paths from source S1 to S4 & S7 so the states are 0,1,and 2. The number of states is 3
# S6 takes one path from source S1 so the number of states of S6 is 2. The combination of states of S8 leads to 0 to 18 states ( 3*3*2=18
#000 001 002 010 011 012 101 102 111 112
S8=np.random.rand(6,18)
r1 = S8.sum(axis=0)
S8 = S8 / r1
#print('S8(l) is',S8[l])
# states of C depends on the parents S7,S8, S9,S10,S11
#S7 takes maximum two paths from source S1 to S7 the number of states of S7 is 3( 0,1,2)
#S8 takes maximum 4 paths from source S1 to S8 the number of states of S8 is 5( 0,1,2,3,4)
#S9 takes maximum one path from source S1 to S9 the number of states of S9 is 2 (0,1)
#S11 takes max 5 paths from source S1 to S11 and the no of states of S11 is 6( 0,1,2,3,4,5)

# the  number of states 3*5*2*6=180

n=np.arange(0,180)


C=np.random.rand(23,)
r2=C.sum(axis=0)
C=C/r2
k=np.arange(0,4)
m=np.arange(0,9)
j=np.arange(0,18)
h=np.arange(0,19)

#S8_index=[f'S8_{i+1}={x}' for i, x in enumerate(l)]
  
end_random=time.time()
print('the computation time to generate CPT is', end_random-start_random)
#000 001 010 011 020 021 100 101 110 111 120 121
'''
G_0=[1,0.3,0.4,0.12,0.4,0.12,0.2,0.06,0.08,0.024,0.08,0.024]
G_1=[0,0.7,0.6,0,0,0,0.8,0,0,0,0,0]
G_2=[0,0,0,0.88,0.6,0,0,0.94,0.92,0,0,0]
G_3=[0,0,0,0,0,0.88,0,0,0,0.976,0.92,0]
G_4=[0,0,0,0,0,0,0,0,0,0,0,0.976]


     
E=[1,2]
def cond_prob(cdash,c,pinhU):
   px_c=1
   pnofail=[]
   sum_pinh=[]
   for j in E:
      
      pnofail.append([0]*len(E))
      sum_pinh.append([0]*len(E))
      print(pnofail[j])
      print(sum_pinh[j])
      for i in range(1,len(pinhU[j])):
          sum_pinh[j]=sum_pinh[j]+pinhU[j][i]
          pnofail[j]=1-sum_pinh[j]
          if (cdash[j]==c[j]):
             pcdash_c[j]=pnofail[j]+sum_pinh[j]
          else:
             pcdash_c[j]=pinhU[j][i]
   for j in range(1,len(E)):
       px_c=px_c*pcdash_c[j]
   return px_c       

pe_c=cond_prob(['B1','C'],['B','C'],pinhU)

print("this is 1-pinhU[5][0]", 1-(pinhU[5][0]))
 values=[C_0,
                              C_1,
                              C_2,
                              C_3,
                              C_4,
                              C_5,
                              C_6,
                              C_7,
                              C_8,
                              C_9,
                              C_10,
                              C_11,
                              C_12,
                              C_13,
                              C_14,
                              C_15,
                              C_16,
                              C_17,
                              C_18,
                              C_19,
                              C_20,
                              C_21,
                              C_22,
                              C_23,
                              C_24,
                              C_25,
                              C_26,
                              C_27,
                              C_28,
                              C_29,
                              C_30,
                              C_31,
                              C_32,
                              C_33,
                              C_34,
                              C_35,
                              C_36]

'''
bayesian_model=BayesianNetwork([('S1','S2'),('S1','S4'),('S2','S4'),('S2','S6'),('S4','S8'),('S6','S8'),('S4','S7'),('S7','S8'),('S6','S9'),('S8','S9'),('S9','S11'),('S7','C'),('S8','C'),('S9','C'),('S10','C'),('S11','C')])

cpd_S1_sn = TabularCPD(variable='S1', variable_card=2, values=[[u0], [1-u0]],state_names={'S1':['zero','one']})
cpd_S3_sn = TabularCPD(variable='S3', variable_card=2, values=[[u0], [1-u0]],state_names={'S3':['zero','one']})
cpd_S5_sn = TabularCPD(variable='S5', variable_card=2, values=[[u0], [1-u0]],state_names={'S5':['zero','one']})



cpd_S2_sn = TabularCPD(variable='S2', variable_card=2,
                      values=[[1,pinhU[1][0]],
                              [0,1-pinhU[1][0]]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1':['zero','one'],
                                   'S2':['zero','one']})



cpd_S4_sn = TabularCPD(variable='S4', variable_card=2,
                      values=[[1,pinhU[1][0]],
                              [0,1-pinhU[1][0]]],
                      evidence=['S1','S2'],
                      evidence_card=[2],
                      state_names={'S1':['zero','one'],
                                   'S2':['zero','one'],
                                   'S4':['zero','one']})
'''
cpd_S3_sn = TabularCPD(variable='S3', variable_card=2,
                      values=[[1,pinhU[2][0]],
                              [0,1-pinhU[2][0]]],
                      evidence=['S2'],
                      evidence_card=[2],
                      state_names={'S2':['zero','one'],
                                   'S3':['zero','one']})




cpd_S5_sn = TabularCPD(variable='S5', variable_card=7,
                      values=S5[v],
                      evidence=['T1','T2'],
                      evidence_card=[4,4],
                      state_names={'T1': ['zero','one','two','three'],
                                   'T2': ['zero','one','two','three'],
                                   'S5':['zero','one','two','three','four','five','six']})
'''

cpd_S6_sn = TabularCPD(variable='S6', variable_card=2,
                      values=[[1,pinhU[2][0]],
                              [0,1-pinhU[2][0]]],
                      evidence=['S2'],
                      evidence_card=[2],
                      state_names={'S2':['zero','one'],
                                   'S6':['zero','one']})

cpd_S9_sn = TabularCPD(variable='S9', variable_card=2,
                      values=[[1,pinhU[4][0]],
                              [0,1-pinhU[4][0]]],
                      evidence=['S6'],
                      evidence_card=[2],
                      state_names={'S6': ['zero','one'],
                                   'S9': ['zero','one']})

cpd_S7_sn = TabularCPD(variable='S7', variable_card=2,
                      values=[[1,pinhU[4][0]],
                              [0,1-pinhU[4][0]]],
                      evidence=['S4'],
                      evidence_card=[2],
                      state_names={'S4': ['zero','one'],
                                   'S7': ['zero','one']})
cpd_S8_sn=TabularCPD(variable='S8',variable_card=18,
                     values=S8[l],
                     evidence=['S4','S6','S7'],
                     evidence_card=[3,2,3],
                     state_names={'S4': ['zero','one','two'],
                                 'S6': ['zero','one'],
                                 'S7': ['zero','one','two']
                                'S8':['zero','one','two','three','four','five']})




cpd_S11_sn = TabularCPD(variable='S11', variable_card=2,
                      values=[[1,pinhU[6][0]],
                              [0,1-pinhU[6][0]]],
                      evidence=['S9'],
                      evidence_card=[2],
                      state_names={'S9': ['zero','one'],
                                   'S11': ['zero','one']})
                       


cpd_C_sn = TabularCPD(variable='C', variable_card=23,
                      values=C[n],
                      evidence=['S7','S8','S9','S10','S11'],
                      evidence_card=[3,6,2,7,6],
                      state_names={'S7': ['zero','one','two'],
                                   'S8': ['zero','one','two','three','four','five'],
                                   'S9':['zero','one','two'],
                                   'S10':['zero','one','two'],
                                   'S11':['zero','one','two','three','four'],
                                   'C':['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen'})

                       


                      # These defined CPDs can be added to the model. Since, the model already has CPDs associated to variables, it will
# show warning that pmgpy is now replacing those CPDs with the new ones.
bayesian_model.add_cpds(cpd_S1_sn,cpd_S2_sn,cpd_S3_sn,cpd_S4_sn,cpd_S5_sn,cpd_S6_sn,cpd_S7_sn,cpd_S8_sn,cpd_S9_sn,cpd_S11_sn,cpd_C_sn)
start=time.time()
belief_propagation = BeliefPropagation(bayesian_model)
new_values=belief_propagation.query(variables=['C'],
                           evidence={'S1':'one'})
end=time.time()
print(new_values)
print('computation time for belief propagation is',end-start)
