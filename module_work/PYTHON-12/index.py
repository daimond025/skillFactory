import json
import pandas as pd
from pprint import pprint

import xml.etree.ElementTree as ET

df_index = ['name', 'price', 'weight', 'class']
df = pd.DataFrame(columns=df_index)

tree = ET.parse('menu.xml')
root = tree.getroot()
for elem in root:
    elements = [elem.get('name'), elem[0].text, elem[1].text, elem[2].text]
    df = df.append(pd.Series(elements, index=df_index), ignore_index=True)
print(df)

# tree = ET.parse('menu.xml')
# root = tree.getroot()
# for elem in root:
#     for subelem in elem:
#         print(elem.attrib['name'], subelem.tag, subelem.text)
# print(len(root[1]))


# input = pd.read_excel("./nakladnaya.xls", header=None, skiprows=2)
# input = input.dropna(axis=0, how='all')
# print(input)
# print(input.iloc[6:8, [0, 2, 6, 9, 11, 12]])

# input = pd.read_excel('http://www.econ.yale.edu/~shiller/data/Fig3-1.xls', header=None)

# data_file = pd.ExcelFile('./Fig3-1.xls')
# input = pd.read_excel(data_file, header=None)
# print(input)

# df = pd.read_csv('recipes.csv')
# ids = list(df['id'])
# ingredients = df.columns.tolist()[3:]
# new_recipes = []
#
# def make_list(row):
#    columns = row.columns.tolist()[3:]
#    ingredients = []
#    for ingredient in columns:
#       if row[ingredient].values[0] == 1:
#          ingredients.append(ingredient)
#    return ingredients
#
# for current_id in ids:
#     cuisine = df[df['id'] == current_id]['cuisine'].iloc[0]
#     current_ingredients = make_list(df[df['id'] == current_id])
#     current_recipe = {'cuisine': cuisine, 'id': int(current_id), 'ingredients': current_ingredients}
#     new_recipes.append(current_recipe)
# with open("new_recipes.json", "w") as write_file:
#    json.dump(new_recipes, write_file)


# with open('recipes.json') as f:
#    recipes = json.load(f)
#
#    ingredients = set()
#    for item in recipes:
#       for ingred in item['ingredients']:
#          ingredients.add(ingred)
#
#    def find_item(cell):
#       if item in cell:
#          return 1
#       return 0
#
#    df = pd.DataFrame(recipes)
#    for item in ingredients:
#       df[item] = df['ingredients'].apply(find_item)
#    df['ingredients'] = df['ingredients'].apply(lambda x: len(x))
#    df.to_csv('recipes.csv', index=False)


   # food = {}
   # for recipe in recipes:
   #    for item in recipe['ingredients']:
   #          food[item] = 0
   # for recipe in recipes:
   #    for item in recipe['ingredients']:
   #       food[item] += 1
   #
   # min_v = 0
   # product = ''
   # for key in food.keys():
   #    if food[key] == 1:
   #       product = key
   #       min_v = food[key]


   # df = pd.DataFrame(food)
   # print(df)

   # ingredients = []
   # for item in recipes:
   #    if item['cuisine'] == 'russian':
   #       for ingred in item['ingredients']:
   #          ingredients.append(ingred)



   # df = pd.DataFrame(ingredients)
   # print(df)