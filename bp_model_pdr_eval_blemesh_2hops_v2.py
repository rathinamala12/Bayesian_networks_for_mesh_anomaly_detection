from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.models import NoisyOrModel
from itertools import product
from itertools import tee                                                                                                             
import pandas as pd
import numpy as np
# simulation for BLE mesh 12 node where each node speaks to client only through 3 hops


# simulation for BLE mesh 12 node where each node speaks to client only through 4 hops

#s1--s4--s7--s10
#| \/| \/ | \/ |
#s2--s5--s8-- c
#| \/| \/ | \/ |
#s3--s6--s9--s11

#1. s1s8-->s8c\
#2. s1s7-->s7c\
#3. s1s9-->s9c\
#4. s1s10-->s10c\
#5. s1s11-->s11c\

'''
u1=np.random.rand()
u2=np.random.rand()
u3=np.random.rand()
u4=np.random.rand()
u5=np.random.rand()
u6=np.random.rand()
u7=np.random.rand()
u8=np.random.rand()
u9=np.random.rand()
u10=np.random.rand()
u11=np.random.rand()
u0=np.random.rand()
'''
q1=np.random.rand()
q2=np.random.rand()
q3=np.random.rand()
q4=np.random.rand()
q5=np.random.rand()
q6=np.random.rand()
q7=np.random.rand()
q8=np.random.rand()
q9=np.random.rand()
q10=np.random.rand()
q0=0
model = NoisyOrModel(['c1', 'c2','c3','c4','c5'], [2, 2, 2, 2, 2 ], [[1-q6,q6],
                                        [1-q7,q7],[1-q8,q8],[1-q9,q9],[1-q10,q10]])


                     
state2=[[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,1,0,0],[0,0,1,0,1],[0,0,1,1,0],[0,0,1,1,1],[0,1,0,0,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,0,1,1],[0,1,1,0,0],[0,1,1,0,1],[0,1,1,1,0],[0,1,1,1,1],[1,0,0,0,0],[1,0,0,0,1],[1,0,0,1,0],[1,0,0,1,1],[1,0,1,0,0],[1,0,1,0,1],[1,0,1,1,0],[1,0,1,1,1],[1,1,0,0,0],[1,1,0,0,1],[1,1,0,1,0],[1,1,0,1,1],[1,1,1,0,0],[1,1,1,0,1],[1,1,1,1,0],[1,1,1,1,1]]
sense=0
def noisyorout5(arr1,arr2,arr3,arr4,arr5):
   return list(product(arr1, arr2,arr3,arr4,arr5))
prob_array5=noisyorout5(model.inhibitor_probability[0],model.inhibitor_probability[1],model.inhibitor_probability[2],model.inhibitor_probability[3],model.inhibitor_probability[4])
#print(prob_array)
#prob_array[3]=np.multiply(model.inhibitor_probability[0],model.inhibitor_probability[1])
prob_array5=np.asarray(prob_array5)
def prob_out1(prob_array,state,sense,size):
    
    noisyor_out0=[]
    noisyor_out1=[]
    for i in range(0,len(prob_array)):
       #print('i is',i)
       l0=1
       l1=0
       a=np.shape(prob_array)

       for j in range(0,size):
           #print((prob_array[i][j]))
           l0=l0* np.power((prob_array[i][j]),state[i][j])
           l1=1-l0    
       #print('l0 is',l0)
       #print('l1 is',l1)
       noisyor_out0.append(l0)
       noisyor_out1.append(l1)
    if sense==0:
              
              return noisyor_out0
    else:
              
              return noisyor_out1

            



s3_0=prob_out1(prob_array5,state2,sense=0,size=5)
#print(s3_0)
s3_1=prob_out1(prob_array5,state2,sense=1,size=5)
#print(s3_1)
# Defining the model structure. We can define the network by just passing a list of edges.
#bayesian_model = BayesianNetwork([('S10','C'), ('S11','C'), ('S8','C'), ('S9','C'),('S7','C'),('S10', 'S7'),('S10', 'S7'),('S10', 'S8'),('S11', 'S8'),('S11', 'S9'),('S11', 'S9'),('S9','S5'),('S9','S6'),('S9','S8'),('S7','S10'),('S4','S7'),('S4','S8'),('S4','S5'),('S5','S8'),('S5','S9'),('S5','S6'),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ('S7','S8'))])#,('S1','HI'),('S2','HI'),('S3','HI'),])
# CPDs can also be defined using the state names of the variables. If the state names are not provided
# like in the previous example, pgmpy will automatically assign names as: 0, 1, 2, ....

bayesian_model=BayesianNetwork([('S1','S7'),('S1','S8'),('S1','S9'),('S1','S10'),('S1','S11'), ('S7','C'),('S8','C'),('S9','C'),('S10','C'),('S11','C')])

cpd_S1_sn = TabularCPD(variable='S1', variable_card=2, values=[[q0], [1-q0]], state_names={'S1': [ 'False','True']})
                             
                     

cpd_S7_sn = TabularCPD(variable='S7', variable_card=2,
                      values=[[q1,0],
                              [1-q1,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S7': ['False','True']})


cpd_S8_sn = TabularCPD(variable='S8', variable_card=2,
                      values=[[q2,0],
                              [1-q2,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S8': ['False','True']})

cpd_S9_sn = TabularCPD(variable='S9', variable_card=2,
                      values=[[q3,0],
                              [1-q3,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S9': ['False','True']})
        
cpd_S10_sn = TabularCPD(variable='S10', variable_card=2,
                      values=[[q4,0],
                              [1-q4,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S10': ['False','True']})
cpd_S11_sn = TabularCPD(variable='S11', variable_card=2,
                      values=[[q5,0],
                              [1-q5,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S11': ['False','True']})      
cpd_C_sn = TabularCPD(variable='C', variable_card=2,
                      values=[s3_0,
                              s3_1],
                      evidence=['S7','S8','S9','S10','S11'],
                      evidence_card=[2,2,2,2,2],
                      state_names={'S7': [ 'False','True'],
                                   'S8':['False','True'],
                                   'S9':['False','True'],
                                   'S10':['False','True'],
                                   'S11':['False','True'],
                                   'C': ['False','True']})


# These defined CPDs can be added to the model. Since, the model already has CPDs associated to variables, it will
# show warning that pmgpy is now replacing those CPDs with the new ones.
bayesian_model.add_cpds(cpd_S1_sn,cpd_S7_sn, cpd_S8_sn, cpd_S9_sn, cpd_S10_sn,cpd_S11_sn,cpd_C_sn)#,cpd_HI_sn)

belief_propagation = BeliefPropagation(bayesian_model)
new_values=belief_propagation.map_query(variables=['C'],
                             evidence={'S1':'True'})
print(new_values)


