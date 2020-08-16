#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:18:42 2020

@author: user
"""


metta=[]
for file in os.listdir('/home/user/Documents/Zeyu/cups-rl2_metadata_collection (manual)/metadata/'):
    if '12' in file:
        print(file)
        with open('/home/user/Documents/Zeyu/cups-rl2_metadata_collection (manual)/metadata/'+file,'rb') as f:
            meta=pickle.load(f) 
            for i in meta:
                met=[]
                for j in i['objects']:
                    temp=list(j.values())
                    for k in range(len(temp)):
                        temp[k]=str(temp[k])
                    #print(temp)
                    met=met+temp
                    #print(met)
                metta.append(met)