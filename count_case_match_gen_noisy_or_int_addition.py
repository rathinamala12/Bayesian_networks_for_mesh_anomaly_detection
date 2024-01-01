import os
import re

d = [1]
a=0
with open('bp_trials_gen_noisy_or_int_addition_1000_1.txt', 'r') as file:
    for line in file:
        #line = line.strip()
        #for word in line.split():
        #node1 = re.match("all is well", line)
        
        
        node1=len(re.findall("{'G': 'four'}", line))
        #node2=len(re.findall("{'C': 'False'}",line))
        #print(node1)
        #node2 = re.match(r"team2.*", word)
        #type(node2)
        if (node1) in d:
            a=a+1
                
    print('percentage of match  of' + "{'G': 'four'}" +'with ground truth is',a)        
            #else:
                #d[node2] = d[node2] + 1

