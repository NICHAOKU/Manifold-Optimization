### -*- coding: utf-8 -*-
##"""
##
##"""
##from docplex.mp.model import Model
##
##import datetime as dt
##import re
##import json
from pulp import *
import pandas as pd
import numpy as np

def transform(filePath):

    shipInfo = pd.read_excel(filePath,sheet_name='Ship')
    tankInfo = pd.read_excel(filePath,sheet_name='Tank')
    pumpInfo = pd.read_excel(filePath,sheet_name='Pump')
    terminal = pd.read_excel(filePath,sheet_name='Terminal')
    maintInfo = pd.read_excel(filePath,sheet_name='Maint')

    jNum = terminal['Number of Jetties'][0]

    Vol = shipInfo['Ship_Volume']
    TArr = shipInfo['Arrival_Time']
    cap = pumpInfo['Capacity']
    tLimit = tankInfo['Capacity']
    shipNum = len(Vol)
    tNum = len(tLimit)
    pNum = len(cap)
    return shipNum, jNum, tNum, pNum, Vol, TArr, cap, tLimit, maintInfo

shipNum, jNum, tNum, pNum, Vol, TArr, cap, tLimit, maintInfo = transform('data.xlsx')

maxTime = 70
minTime = 0
vLimit = 350
M = 100000

W = [LpVariable("W{}{}{}{}{}".format(i,a,b,c,t), 0,1,LpInteger) for i in range(shipNum) for a in range(jNum) for b in range(pNum) for c in range(tNum) for t in range(maxTime)]
X = [LpVariable("X{}{}{}".format(i,a,t), 0,1,LpInteger) for i in range(shipNum) for a in range(jNum) for t in range(maxTime)]
X_hat = [LpVariable("X_hat{}{}{}".format(i,a,t), 0,1,LpInteger) for i in range(shipNum) for a in range(jNum) for t in range(maxTime)]
Y = [LpVariable("Y{}{}{}".format(i,b,t), 0,1,LpInteger) for i in range(shipNum) for b in range(pNum) for t in range(maxTime)]
Z = [LpVariable("Z{}{}{}".format(i,c,t), 0,1,LpInteger) for i in range(shipNum) for c in range(tNum) for t in range(maxTime)]

Tstart = [LpVariable("Tstart{}".format(i), 0,maxTime,LpInteger) for i in range(shipNum)]
Tstop = [LpVariable("Tstop{}".format(i), 0,maxTime,LpInteger) for i in range(shipNum)]
T = [LpVariable("T{}{}".format(i,t), 0,1,LpInteger) for i in range(shipNum) for t in range(maxTime)]
T_hat = [LpVariable("T_hat{}{}".format(i,t), 0,1,LpInteger) for i in range(shipNum) for t in range(maxTime)]

X_x = [LpVariable("X_x{}{}".format(i,a), 0,1,LpInteger) for i in range(shipNum) for a in range(jNum)]
Z_z = [LpVariable("Z_z{}{}".format(i,c), 0,1,LpInteger) for i in range(shipNum) for c in range(tNum)]

W = np.array(W).reshape(shipNum,jNum,pNum,tNum,maxTime)
X = np.array(X).reshape(shipNum,jNum,maxTime)
X_hat = np.array(X_hat).reshape(shipNum,jNum,maxTime)
Y = np.array(Y).reshape(shipNum,pNum,maxTime)
Z = np.array(Z).reshape(shipNum,tNum,maxTime)
T = np.array(T).reshape(shipNum,maxTime)
T_hat = np.array(T_hat).reshape(shipNum,maxTime)
X_x = np.array(X_x).reshape(shipNum,jNum)
Z_z = np.array(Z_z).reshape(shipNum,tNum)


#define the problem
prob = LpProblem("problem", LpMinimize)

# defines the objective function to minimize
prob += pulp.lpSum((Tstop[i]-TArr[i]) for i in range(shipNum))

#define constrains

#constraint 1
for i in range(shipNum):
    prob += Tstart[i] >= TArr[i]+1
    
#constraint 2
for i in range(shipNum):
    for t in range(maxTime):
        for a in range(jNum):
            prob += pulp.lpSum(W[i,a,b,c,t] for b in range(pNum) for c in range(tNum)) >= X[i,a,t]
            prob += pulp.lpSum(W[i,a,b,c,t] for b in range(pNum) for c in range(tNum)) <= X[i,a,t]*M
        for b in range(pNum):
            prob += pulp.lpSum(W[i,a,b,c,t] for a in range(jNum) for c in range(tNum)) == Y[i,b,t]
        for c in range(tNum):
            prob += pulp.lpSum(W[i,a,b,c,t] for b in range(pNum) for a in range(jNum)) >= Z[i,c,t]
            prob += pulp.lpSum(W[i,a,b,c,t] for b in range(pNum) for a in range(jNum)) <= Z[i,c,t]*M

#constraint 3
for i in range(shipNum):
    for a in range(jNum):
        prob += lpSum(X[i,a,t] for t in range(maxTime)) >= X_x[i,a]
        prob += lpSum(X[i,a,t] for t in range(maxTime)) <= X_x[i,a]*M
        for t in range(maxTime):
            prob += X[i,a,t] <= X_hat[i,a,t]
    for c in range(tNum):
        prob += lpSum(Z[i,c,t] for t in range(maxTime)) >= Z_z[i,c]
        prob += lpSum(Z[i,c,t] for t in range(maxTime)) <= Z_z[i,c]*M
        
#constraint 4
for i in range(shipNum):
    for t in range(maxTime):
        prob += pulp.lpSum(X[i,a,t] for a in range(jNum)) == T[i,t]
        prob += pulp.lpSum(Z[i,c,t] for c in range(tNum)) == T[i,t]
        prob += pulp.lpSum(X_hat[i,a,t] for a in range(jNum)) == T_hat[i,t]
        prob += pulp.lpSum(Y[i,b,t] for b in range(pNum)) >= T[i,t]
        prob += pulp.lpSum(Y[i,b,t] for b in range(pNum)) <= T[i,t]*M
        
#constraint 5
for i in range(shipNum):
    prob += pulp.lpSum(X_x[i,a] for a in range(jNum))==1
    prob += pulp.lpSum(Z_z[i,c] for c in range(tNum))==1

#constraint 6
for i in range(shipNum):
    prob += pulp.lpSum(Y[i,b,t]*cap[b] for b in range(pNum) for t in range(maxTime)) >= abs(Vol[i])

#constraint 7
for i in range(shipNum):
    prob += pulp.lpSum(T[i,t] for t in range(maxTime))-Tstop[i]+Tstart[i] <= 1
    prob += pulp.lpSum(T_hat[i,t] for t in range(maxTime))-Tstop[i]+Tstart[i] == 3

#constraint 8
for i in range(shipNum):
    for t in range(maxTime):
        prob += t*T[i,t]+M*(1-T[i,t])-Tstart[i]>=0
        prob += t*T[i,t]-M*(1-T[i,t])-Tstop[i]<=0
        prob += t*T_hat[i,t]+M*(1-T_hat[i,t])-Tstart[i]>=-1
        prob += t*T_hat[i,t]-M*(1-T_hat[i,t])-Tstop[i]<=1

#constraint 9
for t in range(maxTime):
    for a in range(jNum):
        prob += pulp.lpSum(X_hat[i,a,t] for i in range(shipNum))<=1
    for b in range(pNum):
        prob += pulp.lpSum(Y[i,b,t] for i in range(shipNum))<=1
    for c in range(tNum):
        prob += pulp.lpSum(Z[i,c,t] for i in range(shipNum))<=1

#constraint 10
for i in range(shipNum):
    for t in range(maxTime):
        prob += pulp.lpSum(Y[i,b,t]*cap[b] for b in range(pNum)) <= vLimit
        
#constraint 11
for i in range(shipNum):
    for f in range(len(maintInfo)):
        for t in range(maintInfo['Maint_Start_Time'][f],maintInfo['Maint_Stop_Time'][f]):
            if maintInfo['onboarding valve'][f]==-1:
                for a in range(jNum):
                    if maintInfo['offboarding valve'][f]==-1:
                        for c in range(tNum):
                            prob += W[i,a,maintInfo['pump valve'][f],c,t] == 0
                    else:
                        prob += W[i,a,maintInfo['pump valve'][f],maintInfo['offboarding valve'][f],t] == 0
            else:
                for c in range(tNum):
                    prob += W[i,maintInfo['onboarding valve'][f],maintInfo['pump valve'][f],c,t] == 0

#constraint 12
for c in range(tNum):
    for t in range(maxTime):
        prob += pulp.lpSum(W[i,a,b,c,t0]*cap[b]*(int(Vol[i]>0)-int(Vol[i]<0)) for i in range(shipNum) for a in range(jNum) for b in range(pNum) for t0 in range(t))<=tLimit[c]

prob.writeLP("ManifoldOptimization.lp")

prob.solve()
print("Status:", LpStatus[prob.status])
print ("objective=", value(prob.objective))
