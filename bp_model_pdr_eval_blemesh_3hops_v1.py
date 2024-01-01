from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.inference import BeliefPropagation
from pgmpy.models import NoisyOrModel
from itertools import product
from itertools import tee                                                                                                             
import pandas as pd
import numpy as np

# simulation for BLE mesh 12 node where each node speaks to client only through 3 hops


#s1--s4--s7--s10
#| \/| \/ | \/ |
#s2--s5--s8-- c
#| \/| \/ | \/ |
#s3--s6--s9--s11

#1. s1s4-->s4s7-->s7c
#2. s1s5-->s5s8-->s8c
#3. s1s4-->s4s8-->s8c
#4. s1s5-->s5s9-->s9c
#5. s1s5-->s5s7-->s7c

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

model2= NoisyOrModel(['c1', 'c2'], [2, 2], [[1-u4,u4],
                                        [1-u5,u5]])

                     
#print(model2.variables)
#print(model2.inhibitor_probability)
#print(model2.cardinality)


model3 = NoisyOrModel(['c3', 'c4','c5'], [2, 2, 2], [[1-u7,u7],
                                         [1-u8,u8],[1-u9,u9]])

#print(model3.variables)
#print(model3.inhibitor_probability)
#print(model3.cardinality)


def noisyorout(arr1,arr2):
   return list(product(arr1, arr2))
prob_array2=noisyorout(model2.inhibitor_probability[0],model2.inhibitor_probability[1])
#print(prob_array)
#prob_array[3]=np.multiply(model.inhibitor_probability[0],model.inhibitor_probability[1])
prob_array2=np.asarray(prob_array2)  
#(prob_array2)
def noisyorout1(arr1,arr2,arr3):
   return list(product(arr1, arr2,arr3))
prob_array3=noisyorout1(model3.inhibitor_probability[0],model3.inhibitor_probability[1],model3.inhibitor_probability[2])

#prob_array[3]=np.multiply(model.inhibitor_probability[0],model.inhibitor_probability[1])
prob_array3=np.asarray(prob_array3) 
#print(prob_array3)    

state=[[0,0],[0,1],[1,0],[1,1]]
sense=0
state1=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]] 
a=np.shape(prob_array3)
#print(a[1])
def prob_out1(prob_array,state,sense):
    
    noisyor_out0=[]
    noisyor_out1=[]
    for i in range(0,len(prob_array)):
       #print('i is',i)
       l0=1
       l1=0
       a=np.shape(prob_array)

       for j in range(0,a[1]):
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

            
s1_0=prob_out1(prob_array2,state,sense=0)
#print(s1_0)
s1_1=prob_out1(prob_array2,state,sense=1)
#print(s1_1)

s2_0=prob_out1(prob_array3,state1,sense=0)
#print(s2_0)
s2_1=prob_out1(prob_array3,state1,sense=1)
#print(s2_1)

# Defining the model structure. We can define the network by just passing a list of edges.
#bayesian_model = BayesianNetwork([('S10','C'), ('S11','C'), ('S8','C'), ('S9','C'),('S7','C'),('S10', 'S7'),('S10', 'S7'),('S10', 'S8'),('S11', 'S8'),('S11', 'S9'),('S11', 'S9'),('S9','S5'),('S9','S6'),('S9','S8'),('S7','S10'),('S4','S7'),('S4','S8'),('S4','S5'),('S5','S8'),('S5','S9'),('S5','S6'),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ('S7','S8'))])#,('S1','HI'),('S2','HI'),('S3','HI'),])
# CPDs can also be defined using the state names of the variables. If the state names are not provided
# like in the previous example, pgmpy will automatically assign names as: 0, 1, 2, ....

bayesian_model=BayesianNetwork([('S1','S4'),('S4','S7'),('S7','C'),('S1','S5'),('S5','S8'),('S8','C'),('S4','S8'),('S5','S9'),('S9','C'),('S5','S7')])

cpd_S1_sn = TabularCPD(variable='S1', variable_card=2, values=[[u0], [1-u0]], state_names={'S1': [ 'False','True']})

                              
                     

cpd_S4_sn = TabularCPD(variable='S4', variable_card=2,
                      values=[[u1,0],
                              [1-u1,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S4': ['False','True']})
cpd_S5_sn = TabularCPD(variable='S5', variable_card=2,
                      values=[[u1,0],
                              [1-u1,1]],
                      evidence=['S1'],
                      evidence_card=[2],
                      state_names={'S1': [ 'False','True'],
                                   'S5': ['False','True']})

cpd_S7_sn = TabularCPD(variable='S7', variable_card=2,
                      values=[s1_0,
                              s1_1],
                      evidence=['S4','S5'],
                      evidence_card=[2,2],
                      state_names={'S4': [ 'False','True'],
                                   'S5':['False','True'],
                                   'S7':['False','True'],
                                   })                             
cpd_S8_sn = TabularCPD(variable='S8', variable_card=2,
                      values=[s1_0,
                              s1_1],
                      evidence=['S4','S5'],
                      evidence_card=[2,2],
                      state_names={'S4':['False','True'],
                                   'S5':['False','True'],
                                   'S8':['False','True'],
                                   })

cpd_S9_sn = TabularCPD(variable='S9', variable_card=2,
                      values=[[u5,0],
                              [1-u5,1]],
                      evidence=['S5'],
                      evidence_card=[2],
                      state_names={'S5': [ 'False','True'],
                                   'S9': ['False','True']})

cpd_C_sn = TabularCPD(variable='C', variable_card=2,
                      values=[s2_0,
                              s2_1],
                      evidence=['S7','S8','S9'],
                      evidence_card=[2,2,2],
                      state_names={'S7': [ 'False','True'],
                                   'S8':['False','True'],
                                   'S9':['False','True'],
                                   'C': ['False','True']})


# These defined CPDs can be added to the model. Since, the model already has CPDs associated to variables, it will
# show warning that pmgpy is now replacing those CPDs with the new ones.
bayesian_model.add_cpds(cpd_S1_sn, cpd_S4_sn, cpd_S5_sn, cpd_S7_sn,cpd_S8_sn,cpd_S9_sn,cpd_C_sn)#,cpd_HI_sn)

belief_propagation = BeliefPropagation(bayesian_model)
new_values=belief_propagation.map_query(variables=['C'],
                             evidence={'S1':'True'})
print(new_values)


