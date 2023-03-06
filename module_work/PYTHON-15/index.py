import pandas as pd
import csv

from copy import copy
import pandas as pd

def remove_dups(values):
    val = copy(values)
    for i in range(1, len(values)+1):
        print(i)
        exit()
        if values[-i] in values[:-i]:
            val.remove(values[-i])
    return val


print(remove_dups([1, 12, 4, 1, 4, 8]))


# dat = pd.read_csv('input/imdb.csv')
# print(dat.columns)
# print(dat[dat['Title'] == 'Suicide Squad']['Year'])


# def safe_sum(x,y):
#     try:
#         return x + y
#     except TypeError:
#         txt1 = "Can't sum {x} and {y}".format(x=x, y=y)
#         print(txt1)
#         return 0
# safe_sum(5, 'a')



# # TODO 2
# def check_server(mode):
#     if mode == "memory":
#         print('Memory is ok')
#         return 'Memory is ok'
#     elif mode == "connection":
#         print('Connection is ok')
#         return 'Connection is ok'
#     else:
#         txt1 = "My name is {fname}, I'm {age}".format(fname="John", age=36)
#         raise ValueError("")
