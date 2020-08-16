#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:45:51 2020

@author: user
"""

import json

with open('/home/user/Documents/Zeyu/cups-rl4 (without rulebase) (withrest.env)(hardcoded_seeking_agent)(action_recorded)(copy 4RS2hideegg5)/metadataa.json', 'r') as fp:
    dic = json.load(fp)
    
def getList(dic): 
    return list(dic.keys())

obj=dic['objects']
list_to_be_mapped=[]
for i in obj:
    list_to_be_mapped.append(getList(i))
# list_to_be_mapped=getList(dic) 

after=[]
for i in range(len(list_to_be_mapped)):
    after.append(list(map(obj[i].get, list_to_be_mapped[i])))
    
doc=str(after)