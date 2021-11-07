import datetime
import codecs
import os
import time

from bs4 import BeautifulSoup
from lxml.html import fromstring
from itertools import cycle

from requests_ip_rotator import ApiGateway
import logging
import requests
import pandas as pd

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

class Parser:
    def __init__(self, path_file):
        # перменные класса
        self.base_url = 'https://auto.ru/cars/all/?page='
        self.file = None


        #  для избежание повтора  машин
        self.url_cars = []

        # массив для url
        self.model_cars =  'https://auto.ru/cars/all/?page='

        if not os.path.exists(path_file):
            file = codecs.open('./data/Base.csv', 'w+', encoding="utf-8")
            file.write(u'\ufeff')

            #  заголовки
            column = ['url', 'bodyType', 'brand', 'color', 'fuelType', 'modelDate', 'name',
                      'numberOfDoors', 'productionDate', 'vehicleConfiguration',
                      'vehicleTransmission', 'engineDisplacement', 'enginePower',
                      'description', 'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние',
                      'Владельцы', 'ПТС', 'Таможня', 'Владение', 'price', 'start_date',
                      'hidden', 'model', 'racing', 'fuel_rate', 'power_kvt', 'car_class']
            col = ','.join(column)
            col = col + u"\n"
            file.write(col)
            self.file = file
        else:
            data = pd.read_csv('./data/Base.csv',  lineterminator='\n')
            self.url_cars = data['car_url'].tolist()

            file = codecs.open('./data/Base.csv', 'a+', encoding="utf-8")
            self.file = file

    def input_parser(self):
        for i in range(1, 10000):
            self.getCarModelePage(self.model_cars)
            time.sleep(500)

    #  функция получения списка станиц по типу машин
    def getCarModelePage(self,car_model):
        try:
            for i in range(1, 100):
                print('Mодель ' + car_model + ' СТРАНИЦА ' + str(i))
                url_gen = car_model + str(i)

                response = requests.get(url_gen, headers={'charset': 'utf-8'})
                response.encoding = "utf-8"
                html = response.text

                self.get_page_data(html)
                time.sleep(10)
        except Exception:
            print("An exception occurred")
            time.sleep(30)
            return self.getCarModelePage(car_model)


    #  парсинг целой страницы объявлений
    def get_page_data(self, html):
        soup = BeautifulSoup(html, 'lxml')

        cars = soup.find_all('div', class_='ListingItem')
        for car in cars:
            url = car.find('a', class_='ListingItemTitle__link')

            if url is not None and url['href'] != "" and url['href'] not in self.url_cars:
                car_params = self.getCar(url['href'])

                if len(car_params) > 0:

                    car_params = [str(item) for item in car_params]
                    col = ','.join(car_params)
                    col = col + u"\n"
                    self.file.write(col)
                    self.url_cars.append(url['href'])
                    time.sleep(2)

    def get_proxies(self, url):
        url = 'https://free-proxy-list.net/'
        response = requests.get(url)
        parser = fromstring(response.text)
        proxies = set()
        for i in parser.xpath('//tbody/tr')[:100]:
            if i.xpath('.//td[7][contains(text(),"yes")]'):
                # Grabbing IP and corresponding PORT
                proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
                proxies.add(proxy)

        proxy_pool = cycle(proxies)
        url = 'https://httpbin.org/ip'
        proxy_ret = None
        for i in range(1,11):
            proxy = next(proxy_pool)
            proxy_ret = proxy

            try:
                response = requests.get(url, proxies={"http": proxy, "https": proxy})
                print(123)
                print(response.json())
                exit()
            except:
                print(proxy)
                print("Skipping. Connnection error")


    # парсинг отдельной машины - собирание данных
    def getCar(self, url):
        # TODO TMP
        # url = 'https://auto.ru/cars/used/sale/bmw/i3/1105789432-407edd67/'
        print(url)
        # # url = 'https://auto.ru/cars/new/group/hyundai/creta/22913412/22952785/1105604503-694cc2dc/'

        try:
            response = requests.get(url)
            response.encoding = "utf-8"
            car_detail = response.text
            soup = BeautifulSoup(car_detail, 'lxml')

            bodyType = ''
            bodyType_li = soup.find('li', class_='CardInfoRow_bodytype')
            if bodyType_li is not None:
                bodyType = bodyType_li.find('a').text.title()
            else:
                bodyType_li = soup.find('li', class_='CardInfoGroupedRow_bodytype')
                if bodyType_li is not None:
                    bodyType = bodyType_li.find('a').text.title()
            bodyType = self.delcommaSTR(bodyType)

            brand = ''
            brand_elem = soup.find('div', class_='BreadcrumbsPopup')
            if brand_elem is not None:
                brand = brand_elem.find('a').text
            else:
                brand_elem_arr = soup.find_all('li', class_='BreadcrumbsGroup__item')
                if (len(brand_elem_arr) >= 2):
                    brand_elem = brand_elem_arr[2].find('a')
                    if brand_elem is not None:
                        brand = brand_elem.text
            brand = self.delcommaSTR(brand)

            # ЦВЕТ
            color = ''
            color_elem = soup.find('li', class_='CardInfoRow_color')
            if color_elem is not None:
                color = color_elem.find('a').text
            else:
                color_elem = soup.find('li', class_='CardInfoGroupedRow_color')
                if color_elem is not None:
                    color = color_elem.find('a').text
            color = self.delcommaSTR(color)

            fuelType = ''
            fuelType_elem = soup.find('li', class_='CardInfoRow_engine')
            if fuelType_elem is not None:
                fuelType_elem_a = fuelType_elem.find('a')

                if fuelType_elem_a is not None:
                    fuelType = fuelType_elem.find('a').text
                else:
                    fuelType_elem_span = fuelType_elem.find_all('span')
                    if fuelType_elem_span is not None and len(fuelType_elem_span) == 2:
                        fuelType_txt = fuelType_elem_span[1].text.lower().replace("\xa0", "").replace(" ", "")
                        fuelType_txt = fuelType_txt.split('/')
                        if (len(fuelType_txt) == 3):
                            fuelType = fuelType_txt[2]
            else:
                fuelType_elem = soup.find('li', class_='CardInfoGroupedRow_engine')
                if fuelType_elem is not None:
                    fuelType = fuelType_elem.find('a').text
            fuelType = self.delcommaSTR(fuelType)

            # productionDate - год производства
            productionDate = 0
            productionDate_elem = soup.find('li', class_='CardInfoRow_year')
            if productionDate_elem is not None:
                productionDate = productionDate_elem.find('a').text
            else:
                productionDate_elem = soup.find('div', class_='CardHead__year')
                productionDate = productionDate_elem.text
            productionDate = self.delcommaSTR(productionDate)

            #  description - описание
            description = ""
            description_elem = soup.find('div', class_='CardDescriptionHTML')
            if description_elem is not None:
                description_elem_span = description_elem.find_all('span')
                for item in description_elem_span:
                    item_corect = ' ' + item.text.replace("<br>", "").replace(",", " ")
                    description += item_corect
            else:
                description_elem = soup.find('div', class_='CardDescription__textInner ')
                if description_elem is not None:
                    description_elem_span = description_elem.find_all('span')
                    for item in description_elem_span:
                        item_corect = ' ' + item.text.replace("<br>", "").replace(",", " ").replace('\n', " ")
                        description += item_corect
            description = self.delcommaSTR(description)

            # mileage - пробег
            mileage = 0
            mileage_elem = soup.find('li', class_='CardInfoRow_kmAge')
            if mileage_elem is not None:
                mileage_elem_span = mileage_elem.find_all('span')
                if len(mileage_elem_span) == 2:
                    mileage = mileage_elem_span[1].text.replace("км", "").replace("\xa0", "")
            mileage = self.delcommaSTR(mileage)

            # Комплектация - НЕ ПОНЯЛ ЧТО ЗА ИНФА И ОТКУДА ЕЕ С auto.ru БРАТЬ
            complectation = ''

            # Привод
            privod = ''
            privod_elem = soup.find('li', class_='CardInfoRow_drive')
            if privod_elem is not None:
                privod_elem_span = privod_elem.find_all('span')
                if len(privod_elem_span) == 2:
                    privod = privod_elem_span[1].text
            else:
                privod_elem = soup.find('li', class_='CardInfoGroupedRow_drive')
                if privod_elem is not None:
                    privod = privod_elem.find('div', class_='CardInfoGroupedRow__cellValue').text
            privod = self.delcommaSTR(privod)

            # Руль  - gj
            rule = ''
            rule_elem = soup.find('li', class_='CardInfoRow_wheel')
            if rule_elem is not None:
                rule_elem_span = rule_elem.find_all('span')
                if len(rule_elem_span) == 2:
                    rule = rule_elem_span[1].text.lower()
                    if rule == 'левый':
                        rule = 'LEFT'
                    else:
                        rule = 'RIGHT'
            else:
                rule = 'LEFT'

            # Состояние - РЕМОНт - требуется ли ремонт
            status = ''
            status_elem = soup.find('li', class_='CardInfoRow_state')
            if status_elem is not None:
                status_elem = status_elem.find_all('span', class_='CardInfoRow__cell')
                if len(status_elem) == 2:
                    status = status_elem[1].text
            status = self.delcommaSTR(status)

            # Владельцы
            own_count = 1
            own_elem = soup.find('li', class_='CardInfoRow_ownersCount')
            if own_elem is not None:
                own_elem_span = own_elem.find_all('span')
                if len(own_elem_span) == 2:
                    own_count = own_elem_span[1].text.lower().replace("\xa0", "")
                    own_count = ''.join(x for x in own_count if x.isdigit())
            else:
                own_count = 0  # НОВАЯ
            own_count = self.delcommaSTR(own_count)

            # ПТС
            ptc = ''
            ptc_elem = soup.find('li', class_='CardInfoRow_pts')
            if ptc_elem is not None:
                ptc_elem_span = ptc_elem.find_all('span')
                if len(ptc_elem_span) == 2:
                    ptc = ptc_elem_span[1].text.lower()
                    if ptc == 'Оригинал':
                        ptc = 'ORIGINAL'
                    else:
                        ptc = 'DUPLICATE'
            else:
                ptc = 'ORIGINAL'  # НОВАЯ

            # Таможня
            customs = ''
            customs_elem = soup.find('li', class_='CardInfoRow_customs')
            if customs_elem is not None:
                customs_elem_span = customs_elem.find_all('span')
                if len(customs_elem_span) == 2:
                    customs = customs_elem_span[1].text.lower()
                    if customs == 'растаможен':
                        customs = 'TRUE'
                    else:
                        customs = 'FALSE'
            else:
                customs = 'TRUE'  # НОВАЯ

            # Владение
            owner = ''

            # price
            price = ''
            price_elem = soup.find('div', class_='PriceNewOffer__originalPrice')
            if price_elem is not None:
                price = price_elem.text.lower().replace("\xa0", "").replace("₽", "").replace(" ", "").replace(
                    "безскидок",
                    "")
            else:
                price_elem = soup.find('span', class_='OfferPriceCaption__price')
                if price_elem is not None:
                    price = price_elem.text.lower().replace("\xa0", "").replace("₽", "").replace("от", "").replace(" ",
                                                                                                                   "")
            price = self.delcommaSTR(price)

            # start_date -
            start_date = ''
            ct = datetime.datetime.now()
            start_date = ct.timestamp()

            hidden = ''

            #  детальная информация о модели
            url_deteil = soup.find('a', class_='SpoilerLink_type_default')
            response = requests.get(url_deteil['href'])
            response.encoding = "utf-8"
            model_detail = response.text
            soup_detail = BeautifulSoup(model_detail, 'lxml')

            # modelDate - год модели
            modelDate = ''
            modelDate_elem = soup_detail.find('div', class_='search-accordion__header')
            if modelDate_elem:
                modelDate = str(self.getmodeleDate(modelDate_elem))

            name = ''  # Л С
            vehicle = '' # Коробка
            vehicleVol = '' # объем двига
            name_elem_main = soup_detail.find('div', class_='catalog__details-main')
            name_elem_main_dd = name_elem_main.find_all('dd')
            if len(name_elem_main_dd) == 9:
                name = name_elem_main_dd[2].text
                vehicle = name_elem_main_dd[3].text

                vehicleVol = name_elem_main_dd[0].text
                vehicleVol = (vehicleVol.replace("л", "").replace(" ", "").replace(".0", ""))
            elif len(name_elem_main_dd) > 6:
                name = name_elem_main_dd[1].text
                vehicle = name_elem_main_dd[2].text

                vehicleVol = name_elem_main_dd[0].text
                vehicleVol = (vehicleVol.replace("л", "").replace(" ", "").replace(".0", ""))
            elif len(name_elem_main_dd) == 6:  # ЭЛЕКТРО
                name = name_elem_main_dd[1].text
                vehicle = name_elem_main_dd[2].text
                vehicleVol = ''
            name = self.delcommaSTR(name)
            vehicle = self.delcommaSTR(vehicle)
            vehicleVol = self.delcommaSTR(vehicleVol)


            # numberOfDoors - кол-во дверей
            numberOfDoors = 0
            name_elem_main = soup_detail.find_all('div', class_='catalog__column_half')
            general = name_elem_main[4]
            general_div = general.find('div', class_='catalog__details-group')
            if general_div is not None:
                dd = general_div.find_all('dd')
                numberOfDoors = dd[2].text.lower()
            numberOfDoors = self.delcommaSTR(numberOfDoors)

            # vehicleConfiguration - сборная солянка
            vehicleConfiguration = ' '.join([bodyType, vehicle, vehicleVol])

            # vehicleTransmission - тип передачи
            vehicleTransmission = vehicle

            # engineDisplacement - объем двигателя
            engineDisplacement = vehicleVol

            # enginePower - ЛС
            enginePower = name.replace("л.с.", "").replace("Л.С.", "").replace(" ", "")
            enginePower = self.delcommaSTR(enginePower)

            # model
            model = ''
            if len(name_elem_main) > 0:
                model_elem = name_elem_main[0].find('h2')
                model = model_elem.text.replace("Модификация", "").strip()
            model = self.delcommaSTR(model)

            #  разгон
            racing = ''
            if len(name_elem_main) > 0:
                racing_elem = name_elem_main[3].find_all('dd')
                if len(racing_elem) >= 3:
                    racing_txt = racing_elem[2].text.lower()
                    if "с" in racing_txt:
                        racing = racing_elem[2].text.lower().replace("с", "").replace(" ", "")
            racing = self.delcommaSTR(racing)

            # расход
            fuel_rate = ''
            if len(name_elem_main) >= 3:
                fuel_rate_elem = name_elem_main[3].find_all('dd')
                if len(fuel_rate_elem) >= 4:
                    fuel_rate_txt = fuel_rate_elem[3].text.lower()

                    if "л" in fuel_rate_txt:
                        fuel_rate = fuel_rate_txt.replace("л", "").replace(" ", "")
            fuel_rate = self.delcommaSTR(fuel_rate)

            # кВт
            power_kvt = ''
            power_kvt_txt = soup_detail.find('dt', text='Максимальная мощность, л.с./кВт при об/мин')
            if power_kvt_txt is not None:
                power_kvt_txt_parent = power_kvt_txt.parent
                power_kvt_elem_arr = power_kvt_txt_parent.find_all('dd')

                for item_elem in power_kvt_elem_arr:
                    item_elem_txt = item_elem.text.lower()
                    if '/' in item_elem_txt and 'при' in item_elem_txt:
                        power_kvt_arr = item_elem_txt.split('при')
                        if len(power_kvt_arr) == 2 and ('/' in power_kvt_arr[0]):
                            power_kvt_arr = power_kvt_arr[0].split('/')
                            if len(power_kvt_arr) == 2:
                                power_kvt = power_kvt_arr[1]
                        break
            power_kvt = self.delcommaSTR(power_kvt)


            # класс машины
            car_class = ''
            if general_div is not None:
                dd = general_div.find_all('dd')
                car_class = dd[1].text


            row_list = [url, bodyType, brand, color, fuelType, str(modelDate), name, str(numberOfDoors), productionDate,
                        vehicleConfiguration,
                        vehicleTransmission, engineDisplacement, enginePower, description, mileage, complectation,
                        privod,
                        rule, status, own_count, ptc, customs, owner, price, start_date, hidden, model,
                        racing, fuel_rate, power_kvt, car_class]

            row_list = [str(item).replace('\n', '').replace('\r', '').strip() for item in row_list]
            return row_list
        except Exception:
            print("An exception occurred")
            print(Exception)
            time.sleep(50)
            # return self.getCar(url)
            return []

    # удалание зарятых из строки
    def delcommaSTR(self, str_ = ''):
        return str(str_).title().replace(",", ".").replace('\n', " ").strip()

    def getPublishAt(self,str=''):
        mounth = {
            'янв': 1,
            'фев': 2,
            'мар': 3,
            'апр': 4,
            'май': 5,
            'июн': 6,
            'июл': 7,
            'авг': 8,
            'сен': 9,
            'окт': 10,
            'ноя': 11,
            'дек': 12
        }
        date_comp = str.split(' ')

        mounth_num = []
        date_day = 1
        if str != '' and len(date_comp) == 1:
            date_day = int(date_comp[0].replace(" ", ""))
            date_mounth = date_comp[1]
            date_mounth = date_mounth[:3]
            mounth_num = [num for mounth, num in mounth.items() if date_mounth == mounth]

        if str == '' or (len(mounth_num) != 1):
            day = datetime.datetime.now()
            return day.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            day = datetime.datetime(2021, mounth_num[0], date_day, hour=0, minute=0)
            return day.strftime('%Y-%m-%dT%H:%M:%S')

    # функуция получения даты модели
    def getmodeleDate(self,modelDate_elem):
        a_list = modelDate_elem.find_all('a')

        a_curent = a_list[len(a_list) - 2]
        try:
            text_analis = a_curent.text

            pos = text_analis.split('–')
            if len(pos) != 2:
                raise Exception()

            pos_ = pos[0].split(' ')

            modelDate = 0
            for item in reversed(pos_):
                if item.isdigit():
                    modelDate = int(item)
                    break
            return modelDate
        except:
            return ''

path_file = './data/Base.csv'
parser = Parser(path_file)
parser.input_parser()
