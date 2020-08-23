# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:53:48 2020

@author: Lenovo
"""


import shutil
import os
import json

got_issue=[]
for file in os.listdir(r'C:\Users\Lenovo\Desktop\RL\actionet-master\actionet-master\tasks\hide'):
    a=r'C:\Users\Lenovo\Desktop\RL\actionet-master\actionet-master\tasks\hide'+'\\'+file
    if ('json' not in file) and ('processed' not in file):
        f = open(a, "r")
        x=f.read()
        res = x.strip('][').split(', ')
        actions=[]
        objects=[]
        try:
            while ']' not in res[0]:
                res.pop(0)
            res[0]=res[0].split('[', 1)[-1]
    
    #        sep = '|'
            for i in range(len(res)):
    #            res[i] = res[i].split(sep, 1)[0]
                res[i] = res[i].replace("'", "")
            for i in res:
                if i=='MoveAhead':
                    actions.append(0)
                elif i=='MoveBack':
                    actions.append(1)
                elif i=='MoveRight':
                    actions.append(2)
                elif i=='MoveLeft':
                    actions.append(3)
                elif i=='LookUp':
                    actions.append(4)
                elif i=='LookDown':
                    actions.append(5)
                elif i=='RotateRight':
                    actions.append(6)
                elif i=='RotateLeft':
                    actions.append(7)
                elif i=='OpenObject':
                    actions.append(8)
                elif i=='CloseObject':
                    actions.append(9)
                elif i=='PickupObject':
                    actions.append(10)
                elif i=='PutObject':
                    actions.append(11)
                elif i=='MoveHandAhead':
                    actions.append(12)
                elif i=='MoveHandLeft':
                    actions.append(13)
                elif i=='MoveHandRight':
                    actions.append(14)
                elif i=='MoveHandBack':
                    actions.append(15)
                elif i=='MoveHandUp':
                    actions.append(16)
                elif i=='MoveHandDown':
                    actions.append(17)
                elif i=='DropHandObject':
                    actions.append(18)
                elif i=='Crouch':
                    actions.append(19)
                elif i=='Stand':
                    actions.append(20)
#                elif i=='RotateHandX':
#                    actions.append(21)
#                elif i=='RotateHandY':
#                    actions.append(22)
#                elif i=='RotateHandZ':
#                    actions.append(23)
#                else:
                    objects.append(i)
        
            with open(r'C:\Users\Lenovo\Desktop\RL\actionet-master\actionet-master\tasks\hide\processed'+'\\'+file+'steps.json', 'w') as f:
                json.dump(actions, f, indent=0) 
            with open(r'C:\Users\Lenovo\Desktop\RL\actionet-master\actionet-master\tasks\hide\processed'+'\\'+file+'objects.json', 'w') as f:
                json.dump(objects, f, indent=0)
        except:
            pass
        for i in objects:
            if '|' not in i:
                got_issue.append([file,objects])
    
