import requests

# response = requests.get('https://www.cbr-xml-daily.ru/daily_json.js')
# currencies = response.json()

# Задание 1
# def currency_name(currency=''):
#     url = 'https://www.cbr-xml-daily.ru/daily_json.js'
#     response = requests.get(url).json()['Valute']
#     ret = ''
#     for cur in response:
#         if currency == response[cur]['ID']:
#             ret = response[cur]['Name']
#             break
#     return ret
# print(currency_name('R01700J'))


from bs4 import BeautifulSoup
import requests

url = 'https://nplus1.ru/news/2019/06/04/slothbot'
response = requests.get(url)
page = BeautifulSoup(response.text, 'html.parser')
print()
exit()

# url = 'https://nplus1.ru/news/2019/06/04/slothbot'
# response = requests.get(url)
# page = BeautifulSoup(response.text, 'html.parser')
# print(page.title)
# print(page.find('h1').text)
# print(page.find('time').text)

# def wiki_header(url):
#     response = requests.get(url)
#     page = BeautifulSoup(response.text, 'html.parser')
#     h1 = page.find("h1", {"class": "firstHeading"})
#     return h1.text
#
# wiki_header('https://en.wikipedia.org/wiki/Operating_system')

# def get_actors(url):
#     response = requests.get(url)
#     page=BeautifulSoup(response.text, 'html.parser')
#     actor_block = page.find('div', id='actorList')
#     actors = actor_block.find_all('a')
#     actors_list=[]
#     for i in actors[:]:
#         actors_list.append(i.text)
#     return actors_list
# def get_actors(url):
#     response = requests.get(url)
#     page = BeautifulSoup(response.text, 'html.parser')
#     dev = page.find_all('a', class_='styles_link__1dkjp')
#     result = [i.text for i in dev]
#     print(result)
# get_actors('https://www.kinopoisk.ru/film/42326/')

import pandas as pd
from bs4 import BeautifulSoup
import requests

# url = 'https://www.banki.ru/banks/ratings/'
# soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
# all_blocks = soup.find_all('div', class_='layout-column-full')
# data = all_blocks[2].find('table')
# df = pd.read_html(str(data))[0]
# print(df)
# exit()
#
#
# url = 'https://www.cbr.ru/key-indicators/'
# soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
# all_blocks = soup.find_all('div', class_='key-indicator_content offset-md-2')
# data = all_blocks[1].find('table')
# df = pd.read_html(str(data))[0]
# print(df)
# df = pd.read_html(str(data))
# print(df)

token = 'd3b3353bd3b3353bd3b3353b89d3ca8a3cdd3b3d3b3353bb2de9e0bfd4b7fe50e4fa34c'

# url = 'https://api.vk.com/method/users.get'
#
# ids = ",".join(map(str, range(1, 501)))
# params = {'user_ids': ids, 'v': 5.95, 'fields': 'sex', 'access_token': token, 'lang': 'ru'}
# response = requests.get(url, params=params).json()
#
# total = 0
# women = 0
# men = 0
# women_men = 0
# for item in response['response']:
#
#     sex = int(item['sex'])
#     if sex == 1:
#         women += 1
#         total += 1
#     elif sex == 2:
#         men +=1
#         total += 1
#     else:
#         women_men +=1
# t = women / total
# print(women_men)
# print(men)
# print(women)
# print(total)
# print(t)
# print(round(t,2))

# url = 'https://api.vk.com/method/groups.getMembers'
# count = 1000
# offset = 0
# user_ids = []
# max_count = 11966402
# while offset < max_count:
#     print('Выгружаю {} пользователей с offset = {}'.format(count, offset))
#     params = {
#         'group_id': 'vk',
#         'v': 5.95,
#         'count': count,
#         'offset': offset,
#         'access_token': token
#     }
#     # такой же запрос как в прошлый раз
#     r = requests.get(url, params=params)
#     data = r.json()
#     user_ids += data['response']['items']
#
#     if (len(user_ids) > 100000):
#         break
#     # увеличиваем смещение на количество строк выгрузки
#     offset += count
# print(user_ids[99999])

# def get_smm_index(group_name, token):
#     url_1 = 'https://api.vk.com/method/groups.getMembers'
#     params = {
#         'group_id': group_name,
#         'v': 5.95,
#         'access_token': token
#     }
#
#     response_1 = requests.get(url_1, params=params)
#     data = response_1.json()
#     users = data['response']['count']
#
#     url_2 = 'https://api.vk.com/method/wall.get'
#     params = {
#         'domain': group_name,
#         'filter': 'owner',
#         'count': 10,
#         'offset': 0,
#         'access_token': token,
#         'v': 5.95
#     }
#
#     response_2 = requests.get(url_2, params=params)
#     summ_count = 0
#     news = response_2.json()['response']['items'][:]
#     for i in range(10):
#         count = news[i]['comments']['count'] + news[i]['likes']['count'] + news[i]['reposts']['count']
#         summ_count += count
#     return summ_count / users
#
# t = get_smm_index('vk', token)
# print(t)
