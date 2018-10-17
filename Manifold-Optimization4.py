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
    terminal = pd.read_excel(filePath,sheet_name='Terminal')
    return shipInfo, terminal

shipInfo, terminal = transform('data.xlsx')

jNum = terminal['Number of Jetties'][0]
tNum = terminal['Number of Jetties'][0]
pNum = terminal['Number of Pumps'][0]
Vol = shipInfo['Ship_Volume']
TArr = shipInfo['Arrival_Time']
shipNum = len(Vol)
maxTime = 70
minTime = 0
cap = 150
M = 100000

X = [LpVariable("X{}{}{}".format(i,a,t), 0,1,LpInteger) for i in range(shipNum) for a in range(jNum) for t in range(maxTime)]
Y = [LpVariable("Y{}{}{}".format(i,b,t), 0,1,LpInteger) for i in range(shipNum) for b in range(pNum) for t in range(maxTime)]
Z = [LpVariable("Z{}{}{}".format(i,c,t), 0,1,LpInteger) for i in range(shipNum) for c in range(tNum) for t in range(maxTime)]

Tstart = [LpVariable("Tstart{}".format(i), 0,maxTime,LpInteger) for i in range(shipNum)]
Tstop = [LpVariable("Tstop{}".format(i), 0,maxTime,LpInteger) for i in range(shipNum)]
T = [LpVariable("T{}{}".format(i,t), 0,1,LpInteger) for i in range(shipNum) for t in range(maxTime)]

X_x = [LpVariable("X_x{}{}".format(i,a), 0,1,LpInteger) for i in range(shipNum) for a in range(jNum)]
Z_z = [LpVariable("Z_z{}{}".format(i,c), 0,1,LpInteger) for i in range(shipNum) for c in range(tNum)]

X = np.array(X).reshape(shipNum,jNum,maxTime)
Y = np.array(Y).reshape(shipNum,pNum,maxTime)
Z = np.array(Z).reshape(shipNum,tNum,maxTime)
T = np.array(T).reshape(shipNum,maxTime)
X_x = np.array(X_x).reshape(shipNum,jNum)
Z_z = np.array(Z_z).reshape(shipNum,tNum)


#definethe problem
prob = LpProblem("problem", LpMinimize)

# defines the objective function to minimize
prob += pulp.lpSum((Tstop[i]-TArr[i]) for i in range(shipNum))

#define constrains

#constraint 1
for i in range(shipNum):
    prob += Tstart[i] >= TArr[i]
    prob += Tstop[i] >= Tstart[i]
    
#constraint 2
for i in range(shipNum):
    for t in range(maxTime):
        prob += pulp.lpSum(X[i,a,t] for a in range(jNum)) == T[i,t]
        prob += pulp.lpSum(Z[i,c,t] for c in range(tNum)) == T[i,t]
        prob += pulp.lpSum(Y[i,b,t] for b in range(pNum)) >= T[i,t]
        prob += pulp.lpSum(Y[i,b,t] for b in range(pNum)) <= T[i,t]*M

#constraint 3
for i in range(shipNum):
    prob += pulp.lpSum(Y[i,b,t] for b in range(pNum) for t in range(maxTime))*cap >= Vol[i]

#constraint 4
for i in range(shipNum):
    prob += pulp.lpSum(T[i,t] for t in range(maxTime))-Tstop[i]+Tstart[i]== 1

#constraint 5
for i in range(shipNum):
    for t in range(maxTime):
        prob += t*T[i,t]+M*(1-T[i,t])-Tstart[i]>=0
        prob += t*T[i,t]-M*(1-T[i,t])-Tstop[i]<=0

#constraint 6
for t in range(maxTime):
    for a in range(jNum):
        prob += pulp.lpSum(X[i,a,t] for i in range(shipNum))<=1
    for b in range(pNum):
        prob += pulp.lpSum(Y[i,b,t] for i in range(shipNum))<=1
    for c in range(tNum):
        prob += pulp.lpSum(Z[i,c,t] for i in range(shipNum))<=1

#constraint 7
for i in range(shipNum):
    prob += pulp.lpSum(X_x[i,a] for a in range(jNum))==1
    prob += pulp.lpSum(Z_z[i,c] for c in range(tNum))==1
    
#constraint 8
for i in range(shipNum):
    for a in range(jNum):
        prob += lpSum(X[i,a,t] for t in range(maxTime))-X_x[i,a]>=0
        prob += lpSum(X[i,a,t] for t in range(maxTime))-M*X_x[i,a]<=0
    for c in range(tNum):
        prob += lpSum(Z[i,c,t] for t in range(maxTime))-Z_z[i,c]>=0
        prob += lpSum(Z[i,c,t] for t in range(maxTime))-M*Z_z[i,c]<=0

prob.writeLP("ManifoldOptimization.lp")

prob.solve()
print("Status:", LpStatus[prob.status])
print ("objective=", value(prob.objective))
