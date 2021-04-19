import numpy as np # для работы с массивами
import pandas as pd # для работы DataFrame
import seaborn as sns # библиотека для визуализации статистических данных
import matplotlib.pyplot as plt # для построения графиков


# A = np.array([[4,7,-1] ,[-4,2,5],[0,9,4] ])
# b = np.array([[7,5,3]])
# # A_ =np.linalg.inv(A)
#
# print(np.linalg.matrix_rank(A))
# print(np.dot(A_,b.T))

# A = np.array([[1,2] ,[-3,1],[1,2],[1,-1]])
# b = np.array([1,4,5,0])
#
# g = np.dot(A.T,A)
# g_ = np.linalg.inv(g)
#
# bB = np.dot(A.T,b)
# w_ = np.dot(g_,bB)


# A = np.array([[1,1,-1,0] ,[1,1,1,2],[1,0,0,0],[1,2,0,2]])
# A_g = np.dot(A.T,A)

# A = np.array([[11,8]])
# A_centre =  A - (A.mean())
# A_ = A_centre / np.linalg.norm(A_centre)

student = pd.read_csv('Admission_Predict_Ver1.1.csv')

del student['Serial No.']
data = student[student['Research'] == 0][['TOEFL Score', 'CGPA', 'Chance of Admit ']]

Y = data[['Chance of Admit ']]
#
TOEFL=data['TOEFL Score']
CGPA= data['CGPA']

#  без центровки и нормировки
# A= np.column_stack((np.ones(220), TOEFL, CGPA))#
# ww = np.linalg.lstsq(A,Y,rcond=None)
# print(-1.045 * 1 +  0.004 * 107 + 0.148 * 9.1 )
# print(ww)

#  С центровки и нормировки
# TOEFL_с = TOEFL-TOEFL.mean()
# CGPA_с = CGPA-CGPA.mean()
# Y_с = Y-Y.mean()
#
# TOEFL_st = TOEFL_с/np.linalg.norm(TOEFL_с)
# CGP_st = CGPA_с/np.linalg.norm(CGPA_с)
# Y_st = Y_с/np.linalg.norm(Y_с)
#
# A = np.column_stack((TOEFL_st,CGP_st))
# ww = np.linalg.lstsq(A,Y_st,rcond=None)
#
# print(ww)
#
# A_st=np.column_stack(( TOEFL_st, CGP_st))
# w_hat_st = np.linalg.inv(A_st.T@A_st)@A_st.T@Y_st.values
# print(w_hat_st)

F = np.array([[2,1] ,[1,2] ])
u = np.array([1,-1])
v = np.array([1,1])

print(F@u.T)
print(F@v.T)