import numpy as np

import pandas as pd

import math

# x = np.array([4,5,-1])
# t = 200
# y = np.array([2,0,1])
# p= 400
#
# print(math.sqrt(4*4+6*6+1*1))

#
# Hut_Paradise_DF = pd.DataFrame({'1.Rent': [65, 70, 120, 35, 40, 50, 100, 90, 85],
#                                 '2.Area': [50, 52, 80, 33, 33, 44, 80, 65, 65],
#                                 '3.Rooms':[3, 2, 1, 1, 1, 2, 4, 3, 2],
#                                 '4.Floor':[5, 12, 10, 3, 6, 13, 8, 21, 5],
#                                 '5.Demo two weeks':[8, 4, 5, 10, 20, 12, 5, 1, 10],
#                                 '6.Liv.Area': [37, 40, 65, 20, 16, 35, 60, 50, 40]})
#
# A = Hut_Paradise_DF.values

# print(A)
# print(A[:, 3])
# print(len(A[:,0]))

# area = A[:,1]
# live_area = A[:,5]

# rent = A[:,0]
# rent_y = rent * 100 * 4
# print(rent_y/1000)

# week = A[:,4]
# time=np.array([10,20,30,15,5,40,20,8,20])
# print(week@time)


# u=np.array([3,0,1,1,1])
# v=np.array([0,1,0,2,-2])
# w=np.array([1,-4,-1,0,-2])
#
# dd = 2 *v  - 3 * w
# print(u/np.linalg.norm(u))
# print(v/np.linalg.norm(v))
# print(w/np.linalg.norm(w))


# A = np.array([[5,-1,3,1,2], [-2,8,5,-1,1]])
# x = np.array([1,2,3,4,5])


# A=np.array( [ [1,9,8,5] , [3,6,3,2] , [3,3,3,3], [0,2,5,9], [4,4,1,2] ] )
# B=np.array( [ [1,-1,0,1,1] , [-2,0,2,-1,1] ] )

# x = np.array([1,2,1,0,4])
# y = np.array([2,1,-1,1,0])
# z = np.array([-1,1,-1,0,0])
#
#
#
# Count_DF = pd.DataFrame({'Женские стрижки': [10, 2, 12, 4, 6, 10, 22, 7],
#                                 'Мужские стрижки': [5, 21, 12, 8, 25, 3, 1, 0],
#                                 'Окрашивания':[12, 3, 0, 18, 27, 2, 4, 31],
#                               'Укладка':[15, 25, 30, 14, 25, 17, 25, 31],
#                                 'Уход':[10, 6, 4, 5, 18, 12, 20, 28]
#                                 },
#                                index=['Аня', 'Борис', 'Вика', 'Галя', 'Дима', 'Егор', 'Женя','Юра'])
# Price_DF = pd.DataFrame({'Женские стрижки': [2, 1.8, 2, 1.8, 2.5, 5, 1.1, 4.5],
#                                 'Мужские стрижки': [1.5, 2.5, 2, 1.2, 3.5, 5, 1, 4],
#                                 'Окрашивания':[1, 1, 0, 2.8, 2, 3, 1.5, 2.5],
#                               'Укладка':[0.8, 1, 0.5, 0.8, 1, 2, 0.5, 1],
#                                 'Уход':[1, 1, 2, 2, 1.5, 2.5, 1.7, 2]
#                                 },
#                                index=['Аня', 'Борис', 'Вика', 'Галя', 'Дима', 'Егор', 'Женя','Юра'])
#
# C = Count_DF.values
#
# P = Price_DF.values

#  обратная матрица
# A = np.array([[1,2], [2,5]])
# np.linalg.inv(A)

# определитель матрицы
# A = np.array([[1,2], [1,1]])
# B = np.array([[5,-2], [-1,4]])
# Det_a = np.linalg.det(A)
# Detb = np.linalg.det(B)

#  ранг матрицы
# A = np.array([1,2,3,1])
# B = np.array([4,5,6,1])
# C = np.array([7,8,11,1])
# A_ = np.array([A,B,C]).T
# print(np.linalg.matrix_rank(A_))

# A = np.array([[1,0,3,5], [0,4,5,5], [0,0,0,0], [0,0,0,0]])
# B = np.array([[1,0,3,5] , [0,4,5,5], [0,0,0,4], [0,0,0,0]])
#
# print(np.linalg.matrix_rank(A))
# print(np.linalg.matrix_rank(B))

v1 = np.array([9, 10, 7, 7, 9])
v2 = np.array([2, 0, 5, 1, 4])
v3 = np.array([4, 0, 0, 4, 1])
v4 = np.array([3, -4, 3, -1, -4])
A_ = np.array([v1,v2,v3, v4]).T
g = A_.T@A_
print(g)
print(np.linalg.inv(g ))