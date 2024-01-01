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
#| \/| \/ |    |
#s3--s6--s9--s11



#S1----->S4-----S7--XX--S10
# | \   / \ /   \    / 
# |   T1   T3      T5 
# | /   \ /   \   /   \
#S2      S5    S8      C
# | \    / \  /  \    / 
# |   T2    T4     T6  
# | /   \  /  \   /   \  
#S3 ---- S6----S9----- S11



U=['S1','S2','S3','S4','S5','S6','S7','S8','S9','C','S11','T1','T2','T3','T4','T5','T6']

PU=['S1','S1','S2','S1',['T1','T2'],'S3','S4',['T3','T4'],['S6'],['T5','T6'],'S9',['S1','S2','S4'],['S2','S3','S6'],['S4','S5','S7'],['S5','S6','S9'],['S7','S8'],['S8','S9','S11']]  # parent nodes of  U
NP=[[1],[1],[1],[1],[3,3],[1],[1],[8,8],[1],[18,18],[1],[1,1,1],[1,1,1],[1,6,1],[6,1,1],[1,16],[16,1,1]] # no of paths for U
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

#print(u0)  
'''
  pinhU=[[0]*int(NU[j])]*len(U)
  print(pinhU)
'''
inh_prob=[u0,u1,u1,u1,u2,[u1*u1*u1*u1*u3*u3],u2,u2,[u1*u1*u1*u1*u3*u3],u1,[u1*u1*u1*u1*u3],u1,[u1*u1*u3],[u1*u1*u3],[u1*u1*u3],[u1*u1*u3],[u1*u1],[u1*u1*u3]]
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
v=np.arange(0,7)
S5=np.random.rand(7,16)
r=S5.sum(axis=0)
S5=S5/r
l=np.arange(0,17)
#print(l)
S8=np.random.rand(17,81)
r1 = S8.sum(axis=0)
S8 = S8 / r1
#print('S8(l) is',S8[l])
n=np.arange(0,36)
C=np.random.rand(36,18*19)
r2=C.sum(axis=0)
C=C/r2
k=np.arange(0,4)
m=np.arange(0,9)
j=np.arange(0,18)
h=np.arange(0,19)
T1=np.random.rand(4,8)
r3=T1.sum(axis=0)
T1=T1/r3
#print(T1)
T2=np.random.rand(4,8)
r4=T2.sum(axis=0)
T2=T2/r4
T3=np.random.rand(9,28)
r5=T3.sum(axis=0)
T3=T3/r5
T4=np.random.rand(9,28)
r6=T4.sum(axis=0)
T4=T4/r6
T5=np.random.rand(18,2*17)
r7=T5.sum(axis=0)
T5=T5/r7
T6=np.random.rand(19,4*17)
r8=T6.sum(axis=0)
T6=T6/r8
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
bayesian_model=BayesianNetwork([('S1','S2'),('S1','S4'),('S1','T1'),('S2','T1'),('S2','S3'),('S4','T1'), ('S2','T2'),('S3','S6'),('S3','T2'),('T1','S5'),('T2','S5'),('S6','T2'),('S4','T3'),('S4','S7'),('S5','T3'),('S5','T4'),('S7','T3'),('T3','S8'),('T4','S8'),('S5','T4'),('S6','T4'),('S6','S9'),('S9','S11'),('S9','T4'),('S7','T5'),('S8','T5'),('S8','T6'),('S9','T6'),('S11','T6'),('T5','C'),('T6','C')])

cpd_S1_sn = TabularCPD(variable='S1', variable_card=2, values=[[u0], [1-u0]],state_names={'S1':['zero','one']})
cpd_S2_sn = TabularCPD(variable='S2', variable_card=2,
                      values=[[1,pinhU[1][0]],
                              [0,1-pinhU[1][0]]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1':['zero','one'],
                                   'S2':['zero','one']})

cpd_S3_sn = TabularCPD(variable='S3', variable_card=2,
                      values=[[1,pinhU[2][0]],
                              [0,1-pinhU[2][0]]],
                      evidence=['S2'],
                      evidence_card=[2],
                      state_names={'S2':['zero','one'],
                                   'S3':['zero','one']})

cpd_S4_sn = TabularCPD(variable='S4', variable_card=2,
                      values=[[1,pinhU[1][0]],
                              [0,1-pinhU[1][0]]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1':['zero','one'],
                                   'S4':['zero','one']})

cpd_S6_sn = TabularCPD(variable='S6', variable_card=2,
                      values=[[1,pinhU[3][0]],
                              [0,1-pinhU[3][0]]],
                      evidence=['S3'],
                      evidence_card=[2],
                      state_names={'S3':['zero','one'],
                                   'S6':['zero','one']})


cpd_S5_sn = TabularCPD(variable='S5', variable_card=7,
                      values=S5[v],
                      evidence=['T1','T2'],
                      evidence_card=[4,4],
                      state_names={'T1': ['zero','one','two','three'],
                                   'T2': ['zero','one','two','three'],
                                   'S5':['zero','one','two','three','four','five','six']})

cpd_S7_sn = TabularCPD(variable='S7', variable_card=2,
                      values=[[1,pinhU[4][0]],
                              [0,1-pinhU[4][0]]],
                      evidence=['S4'],
                      evidence_card=[2],
                      state_names={'S4': ['zero','one'],
                                   'S7': ['zero','one']})
cpd_S8_sn=TabularCPD(variable='S8',variable_card=17,
                     values=S8[l],
                     evidence=['T3','T4'],
                     evidence_card=[9,9],
                     state_names={'T3': ['zero','one','two','three','four','five','six','seven','eight'],
                                 'T4': ['zero','one','two','three','four','five','six','seven','eight'],
                                 'S8': ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen']})
                             

cpd_S9_sn = TabularCPD(variable='S9', variable_card=2,
                      values=[[1,pinhU[4][0]],
                              [0,1-pinhU[4][0]]],
                      evidence=['S6'],
                      evidence_card=[2],
                      state_names={'S6': ['zero','one'],
                                   'S9': ['zero','one']})



cpd_S11_sn = TabularCPD(variable='S11', variable_card=2,
                      values=[[1,pinhU[6][0]],
                              [0,1-pinhU[6][0]]],
                      evidence=['S9'],
                      evidence_card=[2],
                      state_names={'S9': ['zero','one'],
                                   'S11': ['zero','one']})
                       


cpd_C_sn = TabularCPD(variable='C', variable_card=36,
                      values=C[n],
                      evidence=['T5','T6'],
                      evidence_card=[18,19],
                      state_names={'T5': ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen'],
                                   'T6': ['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen'],
                                   'C':['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen',
                                         'nineteen','twenty','twenty one','twenty two','twenty three','twenty four','twenty five','twenty six','twenty seven','twenty eight','twenty nine','thrity','thirty one','thirty two','thirty three','thirty four' ,'thirty five']})

                       

cpd_T1_sn = TabularCPD(variable='T1', variable_card=4,
                      values=T1[k],
                      evidence=['S1','S2','S4'],
                      evidence_card=[2,2,2],
                      state_names={'S1': ['zero','one'],
                                   'S2': ['zero','one'],
                                   'S4':['zero','one'],
                                   'T1':['zero','one','two','three']})
cpd_T2_sn = TabularCPD(variable='T2', variable_card=4,
                      values=T2[k],
                      evidence=['S2','S3','S6'],
                      evidence_card=[2,2,2],
                      state_names={'S2': ['zero','one'],
                                   'S3': ['zero','one'],
                                   'S6':['zero','one'],
                                   'T2':['zero','one','two','three']})
                      
cpd_T3_sn = TabularCPD(variable='T3', variable_card=9,
                      values=T3[m],
                      evidence=['S4','S5','S7'],
                      evidence_card=[2,7,2],
                      state_names={'S4': ['zero','one'],
                                   'S5': ['zero','one','two','three','four','five','six'],
                                   'S7':['zero','one'],
                                   'T3':['zero','one','two','three','four','five','six','seven','eight']})
cpd_T4_sn=TabularCPD(variable='T4',variable_card=9,
                     values=T4[m],
                     evidence=['S5','S6','S9'],
                     evidence_card=[7,2,2],
                     state_names={'S5': ['zero','one','two','three','four','five','six'],
                                  'S6': ['zero','one'],
                                  'S9':['zero','one'],
                                  'T4':['zero','one','two','three','four','five','six','seven','eight']})

cpd_T5_sn=TabularCPD(variable='T5', variable_card=18,
                      values=T5[j],
                      evidence=['S7','S8'],
                       evidence_card=[2,17],
                       state_names={'S7':['zero','one'],
                                     'S8':['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen'],
                                     'T5':['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen']})

cpd_T6_sn=TabularCPD(variable='T6', variable_card=19,
                      values=T6[h],
                       evidence=['S8','S9','S11'],
                       evidence_card=[17,2,2],
                       state_names={'S9':['zero','one'],
                                     'S8':['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen'],
                                    'S11':['zero','one'],
                                     'T6':['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen']})


                      # These defined CPDs can be added to the model. Since, the model already has CPDs associated to variables, it will
# show warning that pmgpy is now replacing those CPDs with the new ones.
bayesian_model.add_cpds(cpd_S1_sn,cpd_S2_sn,cpd_S3_sn,cpd_S4_sn,cpd_S5_sn,cpd_S6_sn,cpd_S7_sn,cpd_S8_sn,cpd_S9_sn,cpd_S11_sn,cpd_C_sn,cpd_T1_sn,cpd_T2_sn,cpd_T3_sn,cpd_T4_sn,cpd_T5_sn,cpd_T6_sn)
start=time.time()
belief_propagation = BeliefPropagation(bayesian_model)
new_values=belief_propagation.query(variables=['C'],
                           evidence={'S7':'one','T5':'one'})
end=time.time()
print(new_values)
print('computation time for belief propagation is',end-start)
