import statistics

class Dog():
    def bark(self):
        return "Bark!"
    def give_paw(self):
        return "Paw"
# class OwnLogger():
#     def __init__(self):
#         self.dict =  {
#             "info": [],
#             "warning": [],
#             "error": [],
#             "all": [],
#         }
#     def log(self, message, level):
#         self.dict[level].append(message)
#         self.dict['all'].append(message)
#
#     def show_last(self, level = 'all'):
#         if self.dict[level]:
#             return  self.dict[level][-1]
#         else:
#             return None

logger = OwnLogger()
logger.log("System started", "info")
print(logger.show_last("error"))
logger.log("Connection instable", "warning")
logger.log("Connection lost", "error")

print(logger.show_last())
print(logger.show_last("info"))



# class IntDataFrame():
#     def __init__(self, column, fill_value=0):
#         self.column = column
#         self.fill_value = fill_value
#         self.fill_missed()
#         self.to_int()
#
#     def fill_missed(self):
#         for i, value in enumerate(self.column):
#             if value is None or value == '':
#                 self.column[i] = self.fill_value
#
#     def to_int(self):
#         self.column = [int(value) for value in self.column]
#
#     def count(self):
#         acamulate = []
#         for  i, value in enumerate(self.column):
#             if value != 0:
#                 acamulate.append(value)
#         return len(acamulate)
#
#     def unique(self):
#         acamulate = set()
#         for i, value in enumerate(self.column):
#             acamulate.add(value)
#         return len(acamulate)

# df = IntDataFrame([4.7, 4, 3, 0, 2.4, 0.3, 4])
# print(df.count())
# print(df.unique())

            # class User():
#     def __init__(self, email, password , balance):
#         self.email = email
#         self.password = password
#         self.balance = balance
#     def login(self, email, password):
#         if self.email == email and self.password == password:
#             return True
#         else:
#             return False
#     def update_balance(self,amount):
#         self.balance += amount
#
# user = User("gosha@roskino.org", "qwerty", 20_000)
# print( user.login("gosha@roskino.org", "qwerty123"))
# print(user.login("gosha@roskino.org", "qwerty"))
# user.update_balance(200)
# user.update_balance(-500)
# print(user.balance)

# class DepartmentReport():
#     def __init__(self, company_name):
#         self.revenues = []
#         self.company_name = company_name
#
#     def add_revenue(self, amount):
#         self.revenues.append(amount)
#
#     def average_revenue(self):
#         return 'Average department revenue for ' + self.company_name + ": " + str(int(sum(self.revenues)/len(self.revenues)))
#
#
# report = DepartmentReport("Danon")
# report.add_revenue(1_000_000)
# report.add_revenue(400_000)

# print(report.average_revenue())
