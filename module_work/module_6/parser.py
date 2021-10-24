from bs4 import BeautifulSoup
import requests
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

f = open('./data/Base.csv', 'w')
# ЗАГОЛОВКИ ДЛЯ ДАННЫх
file = open('./data/Base.csv', 'a+', encoding="utf-8")
column = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate', 'name',
       'numberOfDoors', 'productionDate', 'vehicleConfiguration',
       'vehicleTransmission', 'engineDisplacement', 'enginePower',
       'description', 'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние',
       'Владельцы', 'ПТС', 'Таможня', 'Владение', 'price', 'start_date',
       'hidden', 'model']
col = ','.join(column)
file.write(col + "\n")

def get_page_data(html):
    soup = BeautifulSoup(html, 'lxml')

    cars = soup.find_all('div', class_='ListingItem')
    for car in cars:
        url = car.find('a', class_='ListingItemTitle__link')

        if len(url) > 0 and url['href'] != "":
            getCar(url['href'])


def getCar(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    car_detail = response.text
    soup = BeautifulSoup(car_detail, 'lxml')

    bodyType = ''

    print(url)
    exit()
    bodyType_li = soup.find('li', class_='CardInfoRow_bodytype')
    if bodyType_li.find('a'):
        bodyType = bodyType_li.find('a').text.title()

    brand = ''
    brand_elem = soup.find('div', class_='BreadcrumbsPopup')
    if brand_elem.find('a'):
        brand = brand_elem.find('a').text.title()

    color = ''
    color_elem = soup.find('li', class_='CardInfoRow_color')
    if color_elem.find('a'):
        color = color_elem.find('a').text.title()

    fuelType = ''
    fuelType_elem = soup.find('li', class_='CardInfoRow_engine')
    if fuelType_elem.find('a'):
        fuelType = fuelType_elem.find('a').text.title()

    # modelDate
    modelDate = ''
    modelDate_elem = soup.find('li', class_='CardInfoRow_year')
    if modelDate_elem.find('a'):
        modelDate = modelDate_elem.find('a').text.title()

    # name
    name = ''

    #  детальная информация о моделиф
    url_deteil = soup.find('a', class_='SpoilerLink_type_default')
    response = requests.get(url_deteil)
    response.encoding = "utf-8"
    model_detail = response.text




# browser = webdriver.Firefox(executable_path=r'./selenium_driver/geckodriver.exe')
base_url = 'https://auto.ru/cars/all/?page='
for i in range(1, 100):  # Number of pages you want to parse. In this case first 9 pages
    url_gen = base_url + str(i)

    response = requests.get(url_gen, headers={'charset': 'utf-8'})
    response.encoding = "utf-8"
    html = response.text

    model, desc, price, year, km = get_page_data(html)
    # write_csv(f, model, desc, price, year, km)

file.close()

