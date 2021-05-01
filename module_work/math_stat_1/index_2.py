import numpy as np

Husband_Income = np.array([100,220,140])
Wife_Income = np.array([150,200,130])
Mother_In_Law_Income = np.array([90,80,100])

Husband_Сonsumption = np.array([50,50,60])
Wife_Сonsumption = np.array([100,80,140])
Mother_In_Law_Сonsumption = np.array([100,20,140])


Inc= np.array([Husband_Income,Wife_Income,Mother_In_Law_Income]).T

print(Inc)
Inc_a = Inc* 0.87


Cons=np.array([Husband_Сonsumption,Wife_Сonsumption,Mother_In_Law_Сonsumption]).T
dd =  Inc_a - Cons
print(dd)
