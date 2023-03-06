import pandas as pd
import numpy as np
import datetime as datetime
import locale
locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')

import requests
from bs4 import BeautifulSoup

url = 'http://chgk.tvigra.ru/letopis/?2016'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'lxml')
table = soup.find_all("table", {"class":"txt"})
tr = table[0].findChildren("tr")

monthDict = {
    'марта':3,
    'апреля':4,
    'мая':5,
    'июня':6,
    'сентября':9,
    'сентября':9,
}

for item in tr[1].findChildren("a"):
    # sdf = datetime.datetime.strptime(item.getText(), "%d %B").date()
    print(item.getText())
    exit()
