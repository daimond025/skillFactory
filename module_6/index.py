import datetime

import numpy as np
import pandas as pd
import sys
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import xgboost as xgb
import json
from pycbrf.toolbox import ExchangeRates, Banks
import lazypredict
from lightgbm import LGBMRegressor

from lazypredict.Supervised import LazyRegressor
# from pandas_profiling import ProfileReport
from scipy.stats import ttest_ind
from itertools import combinations
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.ensemble import StackingRegressor

warnings.simplefilter('ignore')
sns.set()

# Неизвестные значения
UNKNOWN_VAL = -1
# Неизвестные строка
UNKNOWN_STR = 'UNKNOWN'

# Состояние  значение по умолчанию
repair_state_no = 'repair_no'
repair_state_yes = 'repair_yes'


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def differenceTrainTest(test, train, col):
    model_diff = set(test[col].to_list()) - set(train[col].to_list())
    print('Количество значений признак нет в тренировочном датасете' + str(len(model_diff)))
    print(model_diff)


def Column_lower_strip(data_1=None, data_2=None, col=''):
    """
     Функция предобработк - удаление пробелом по бокам
    """
    if data_1 is not None:
        data_1[col] = data_1[col].str.lower()
        data_1[col] = data_1[col].str.strip()
    if data_2 is not None:
        data_2[col] = data_2[col].str.lower()
        data_2[col] = data_2[col].str.strip()

    if (data_1 is not None) and (data_2 is not None):
        return data_1, data_2
    elif data_1 is not None:
        return data_1


def reDefineColor(data, col):
    """
        Переопределение цвета
    :param data:
    :param col:
    :return data:
    """
    color_dict = {'040001': 'чёрный',
                  '40001': 'чёрный',
                  'EE1D19': 'красный',
                  '0000CC': 'синий',
                  'CACECB': 'серебристый',
                  '007F00': 'зелёный',
                  'FAFBFB': 'белый',
                  '97948F': 'серый',
                  '22A0F8': 'голубой',
                  '660099': 'пурпурный',
                  '200204': 'коричневый',
                  'C49648': 'бежевый',
                  'DEA522': 'золотистый',
                  '4A2197': 'фиолетовый',
                  'FFD600': 'жёлтый',
                  'FF8649': 'оранжевый',
                  'FFC0CB': 'розовый'}
    data[col].replace(to_replace=color_dict, inplace=True)
    return data


# TODo ANALIZ
def getEmptyValue(data, col):
    """
     Функция получение информации о пропусках или пустых значения категориальнх признаков
    """
    train = data.query('sample == 1')
    test = data.query('sample == 0')

    x = train[col].unique()
    empty_x = (train[col] == '').sum()
    print('Количество уникальных значений для train ' + str(len(x)))
    print('Количество пропусков значений   для train ' + str(empty_x))

    y = test[col].unique()
    empty_y = (test[col] == '').sum()
    print('Количество уникальных значений для test ' + str(len(y)))
    print('Количество пропусков значений   для test ' + str(empty_y))


# TODo ANALIZ
def definePopularColumn(data, col, sample=1):
    """
    Определяет 10 самых популярных значений
    """
    train = data.query('sample == ' + str(sample))

    popular = train[col].value_counts().head(10).to_dict()
    return popular


# TODo ANALIZ
def defineDiffPopularCol(data, col):
    """
        определение разницы значений признаков
    :param data:
    :param col:
    :return:
    """

    train = data.query('sample == 1')
    train_val = train[col].value_counts().to_dict()

    test = data.query('sample == 0')
    test_val = test[col].value_counts().to_dict()

    print("Значений которых нет в test Но есть в train")
    print(set(train_val) - set(test_val))
    print("Значений которых нет в train Но есть в test")
    print(set(test_val) - set(train_val))


# TODo ANALIZ
def get_boxplot(data, column):
    """
    Функция для отрисовки коробочной диаграммы для нечисловых признаков.

    На вход получаем колоноку, для которой строим график.
    График отрисовываем относительно целевой переменной pricing, ограниченной по квантилю.
    """
    fig, ax = plt.subplots(figsize=(25, 5))
    plt.subplots_adjust(wspace=0.5)
    sns.boxplot(x=column, y='price',
                data=data,
                ax=ax)
    plt.xticks(rotation=45)
    # поскольку в price много выбросов, огриничимся 75% квантилем
    ax.set_ylim(0, (data.price.quantile(0.75) + 8 *
                    (data.price.quantile(0.75) - data.price.quantile(0.25))))
    ax.set_title('Boxplot для ' + column, fontsize=20)
    plt.show()


# TODo ANALIZ
def analiz_catigarial_feature(data, col, sample=1):
    """
        Гиисрограмма признаков - категориадльных
    :param data:
    :param col:
    :param sample:
    :return:
    """

    if sample == 0 or sample == 1:
        train = data.query('sample == ' + str(sample))
    else:
        train = data

    fig, axes = plt.subplots(figsize=(30, 8))

    mean = train[col].value_counts().values.mean()
    x = train[col].unique()

    if sample == 0 or sample == 1:
        sns.histplot(data=train, x=train[col],
                     stat='count', bins=data[col].nunique())
    else:
        sns.histplot(data=train, x=train[col],
                     stat='count', bins=data[col].nunique(), hue='sample')

    axes.plot(x, [mean for i in x], '--', color='r')
    plt.xticks(rotation=45)
    plt.xlabel(col, fontsize=20)
    plt.ylabel('Количество', rotation='horizontal')

    plt.title('Распределение количества объясвлений по ' + col, fontsize=20)
    fig.suptitle('Общее количество элементов ' + str(len(x)), fontsize=22, fontweight='bold')
    plt.show()

    # Строим боксплот
    get_boxplot(train, col)


def getSellIdModel_fromURL(data):
    """
        Функция получения МОДЕЛИ И ИД ПРОДАВЦА ИЗ URL МАШИНЫ
        ИЛИ если url датасете нет - ставим значение по умолчанию
    """
    if ('car_url' in data.columns) and ('model' not in data.columns):
        data['model'] = str(data['car_url'].str.split('/').str.get(7).str.strip()).lower()

        dict = {
            r'\n[0-9]*': ''

        }
        data['model'].replace(regex=dict, inplace=True)
        # data['model'] = re.sub(r'\n[0-9]', '', data['model'] )

    elif 'model' not in data.columns:
        data['model'] = UNKNOWN_STR

    if ('car_url' in data.columns) and ('sell_id' not in data.columns):
        data['sell_id_arr'] = data['car_url'].str.split('/').str.get(-2).str.strip()
        data['sell_id'] = data['sell_id_arr'].str.split('-').str.get(0).str.strip()

        data.drop('sell_id_arr', axis=1, inplace=True)
    else:
        data['sell_id'] = UNKNOWN_VAL
    return data


def replaceBodyType(data_1, data_2=None, col=''):
    """
     Функция унификации типов кузовов
    :param data_1:
    :param data_2:
    :param col:
    :return:
    """
    dic = {
        r'внедорожник 3 дв.*': 'внедорожник 3 дв.',
        r'внедорожник 5 дв.*': 'внедорожник 5 дв.',
        r'внедорожник открытый.*': 'внедорожник открытый',
        r'кабриолет.*': 'кабриолет',
        r'компактвэн.*': 'компактвэн',
        r'купе.*': 'купе',
        r'купе-хардтоп.*': 'купе-хардтоп',
        r'лимузин.*': 'лимузин',
        r'лифтбек.*': 'лифтбек',
        r'микровэн.*': 'микровэн',
        r'минивэн.*': 'минивэн',
        r'пикап двойная кабина.*': 'пикап двойная кабина',
        r'пикап одинарная кабина.*': 'пикап одинарная кабина',
        r'пикап полуторная кабина.*': 'пикап полуторная кабина',
        r'родстер.*': 'родстер',
        r'седан \d+.*': 'седан 2 дв.',
        r'седан-хардтоп.*': 'седан-хардтоп',
        r'седан [a-zA-Z]+.*': 'седан',
        r'тарга.*': 'тарга',
        r'универсал 5 дв.*': 'универсал 5 дв.',
        r'фастбек.*': 'фастбек',
        r'фургон.*': 'фургон',
        r'хэтчбек 3 дв.*': 'хэтчбек 3 дв.',
        r'хэтчбек 4 дв.*': 'хэтчбек 4 дв.',
        r'хэтчбек 5 дв.*': 'хэтчбек 5 дв.'}

    data_1[col].replace(regex=dic, inplace=True)
    if data_2 is not None:
        data_2[col].replace(regex=dic, inplace=True)
    if data_2 is None:
        return data_1
    else:
        return data_1, data_2


def trainDefineState(data_1, data_2=None, col=''):
    """
     Переопределение параметров состояния постави по умолдчанию не требует ремонта
    :param col:
    :return:
    """

    data_1[col].fillna(repair_state_no, inplace=True)
    if data_2 is not None:
        data_1[col].fillna(repair_state_no, inplace=True)

    replace_item = {
        'не требует ремонта': repair_state_no,
        'битый / не на ходу': repair_state_yes,
    }

    # data_1[col].replace(to_replace=replace_item, inplace=True)
    data_1 = data_1.replace({col: replace_item})

    if data_2 is not None:
        data_2 = data_2.replace({col: replace_item})
    if data_2 is None:
        return data_1
    else:
        return data_1, data_2


def json_to_series(row, params, some):
    super_gen = json.loads(row['super_gen'])
    row['vehicleTransmission'] = super_gen['transmission']
    row['drivertrain'] = super_gen['gear_type']
    if 'acceleration' in super_gen:
        row['racing'] = super_gen['acceleration']
    else:
        row['racing'] = -1
    return row


def testTransmission(data):
    """
        Функция вытаскивания признака тип передачи из тестового набора
    :param data:
    :param col:
    :return:
    """
    data = data.apply(json_to_series, args=([], 1), axis=1)
    data['vehicleTransmission'].replace({
        r'AUTOMATIC': 'Автомат', r'MECHANICAL': 'Механика',
        r'VARIATOR': 'Вариатор', r'ROBOT': 'Робот'}, inplace=True)

    data['drivertrain'].replace({
        r'FORWARD_CONTROL': 'передний', r'ALL_WHEEL_DRIVE': 'полный',
        r'REAR_DRIVE': 'задний'}, inplace=True)
    return data


def testDriveSide(data, col):
    """
         Функция замены стороны руля - и унификация признака
    :param data:
    :param col:
    :return:
    """
    data[col].replace({
        r'левый': 'LEFT', r'правый': 'RIGHT'}, inplace=True)
    return data


def testOnwerCounts(data_1, col=''):
    """
        Унификация признака кол-во владельцев
    :param data_1:
    :param col:
    :return:
    """
    data_1[col].replace({
        '1 владелец': 'one', "2 владельца": 'two', "3 или более": 'three more'}, inplace=True)

    return data_1


def testPTS(data_1, col=''):
    """
        Унификация признака PTS
    :param data_1:
    :param col:
    :return:
    """
    data_1[col].replace({
        "оригинал": "ORIGINAL", "Оригинал": "ORIGINAL", "дубликат": "DUPLICATE"}, inplace=True)
    return data_1


def testСustoms(data_1, col=''):
    """
        Унификация признака PTS
    :param data_1:
    :param col:
    :return:
    """
    data_1[col].replace({
        "растаможен": "TRUE"}, inplace=True)
    return data_1


def trainVehicleTransmission(data, col):
    """
        Функция переименовывание привода
    :param data:
    :param col:
    :return data:
    """
    data[col].replace({
        r'AUTOMATIC': 'Автомат', r'MECHANICAL': 'Механика',
        r'VARIATOR': 'Вариатор', r'ROBOT': 'Робот'}, inplace=True)
    return data


def reDefineTrainEnginDisplacement(row, params, some):
    name = str(row['name'])
    volume = re.findall('\d\.\d', name)

    value_define = -1
    if len(volume) >= 1:
        value_define = volume[0]

    row['engineDisplacement'] = value_define
    return row


def trainEnginEDisplacement(data, col):
    """
        В тренивочном наборе отсутувет значение - берем его из поля name
    :param data:
    :param col:
    :return train:
    """
    data = data.apply(reDefineTrainEnginDisplacement, args=([], 1), axis=1)
    return data[data[col] != -1]


def trainOnwerCounts(data_1=None, data_2=None, col=''):
    data_1[col] = data_1[col].astype(int)

    data_1[col].replace({
        0: 'non', 1: 'one', 2: 'two', 3: 'three more'}, inplace=True)

    data_2[col].replace({
        0: 'non', 1: 'one', 2: 'two', 3: 'three more'}, inplace=True)
    return data_1, data_2


def preColumnToInt(data_1, data_2=None, col=''):
    """
     Функция преобразования к числу + проверка
    :param data_1:
    :param data_2:
    :param col:
    :return:
    """
    # data_1 = data_1[data_1[col].apply(lambda x: str(x).isdigit())]
    data_1[col] = data_1[col].astype(int)

    if data_2 is not None:
        # data_2 = data_2[data_2[col].apply(lambda x: str(x).isdigit())]
        data_2[col] = data_2[col].astype(int)

    if data_2 is None:
        return data_1
    else:
        return data_1, data_2


def preColumnToFloat(data_1, data_2, col):
    data_1[col] = data_1[col].astype(float)
    data_2[col] = data_2[col].astype(float)
    return data_1, data_2


def redefineDateToInix(data, col):
    """
        Функция удаление  незапролненных строк + выстаскивание даты  из даты год
    :param data:
    :param col:
    :return:
    """
    data.dropna(subset=[col], inplace=True)

    data[col] = pd.to_datetime(data[col], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    data['start_year'] = data[col].dt.year
    data['start_day'] = data[col].dt.strftime('%Y-%m-%d')
    return data


def reDefineTimstampToDate(data, drob=False, col=''):
    """
        Функция удаление незапролненных строк + выстаскивание даты  из unixtimeStamp
    :param drob:
    :param data:
    :param col:
    :return:
    """
    if drob:
        data.dropna(subset=[col], inplace=True)
    data[col] = pd.to_datetime(data[col], unit='s')
    data['start_year'] = data[col].dt.year
    data['start_day'] = data[col].dt.strftime('%Y-%m-%d')
    return data


def defineRacing(date_1, date_2, col):
    """
        Функция определяет время разгона - так как значения в готовом train нет - ставим значение по умолчанию
    :param date_1:
    :param date_2:
    :param col:
    :return:
    """
    # по умолчанию поставим
    date_1[col] = UNKNOWN_VAL
    date_2.dropna(subset=[col], inplace=True)
    return date_1, date_2


# Алвавт всех  производителей
vendor_dic = {}
vendor_dic['EUROPEAN'] = ['PORSCHE', 'LANDROVER', 'JAGUAR',
                          'MINI', 'RENAULT', 'OPEL', 'PEUGEOT', 'CITROEN', 'LAND ROVER', 'ROLLS-ROYCE', 'VOLKSWAGEN',
                          'MERCEDES-BENZ', 'BMW', 'FERRARI', 'AUDI', 'LAMBORGHINI', 'VOLVO', 'SKODA', 'FIAT', 'BENTLEY',
                          'VORTEX', 'MASERATI', 'SMART', 'MATRA', 'ALFAROMEO', 'SEAT', 'TATRA', 'DACIA', 'MAYBACH',
                          'MERCEDES', 'ROVER', 'SAAB', 'EXEED', 'BUGATTI', 'ASTONMARTIN', 'LANCIA']
vendor_dic['RUSSIAN'] = ['LADA(ВАЗ)', 'УАЗ', 'ГАЗ', 'ЗАЗ', 'ИЖ', 'МОСКВИЧ', 'ТАГАЗ', 'DONINVEST', 'DERWAYS']
vendor_dic['ARABIC'] = ['IRANKHODRO']
vendor_dic['JAPANESE'] = ['SUBARU', 'MAZDA', 'SUZUKI', 'LEXUS', 'TOYOTA', 'NISSAN', 'INFINITI', 'MITSUBISHI', 'HONDA',
                          'DATSUN', 'ISUZU', 'ACURA', 'DAIHATSU']
vendor_dic['AMERICAN'] = ['CHEVROLET', 'CHRYSLER', 'CADILLAC', 'JEEP', 'FORD', 'DODGE', 'RAM', 'PONTIAC', 'HUMMER',
                          'LINCOLN', 'SATURN', 'BUICK', 'TESLA', 'MERCURY', 'GMC']
vendor_dic['ASIAN'] = ['HYUNDAI', 'DAEWOO', 'KIA', 'CHERY', 'SSANG YONG', 'GEELY', 'GREATWALL', 'LIFAN', 'SSANGYONG',
                       'FAW', 'GENESIS', 'HAVAL', 'DONGFENG', 'JAC', 'RAVON', 'CHANGAN', 'BYD', 'ZX', 'HAIMA', 'DADI',
                       'BRILLIANCE', 'LUXGEN', 'ZOTYE']


def defineVendorRow(row, params, some):
    find = False
    search_brand = str(row['brand']).replace('_', '').replace(' ', '').upper().strip()
    for vendor, brand in vendor_dic.items():
        if search_brand in brand:
            row['vendor'] = vendor
            find = True
            break
    if not find:
        print('Не нашел производителя для марки ')
        print(row['brand'])
        print(search_brand)
        row['vendor'] = ''
    return row


def defineVendor(data_1, data_2, col):
    """
        Функция определения страны производителя
    :param data_1:
    :param data_2:
    :param col:
    :return:
    """
    data_1.dropna(subset=[col], inplace=True)
    data_2.dropna(subset=[col], inplace=True)

    data_1 = data_1.apply(defineVendorRow, args=([], 1), axis=1)
    data_2 = data_2.apply(defineVendorRow, args=([], 1), axis=1)
    return data_1, data_2


def UseCourseDate(row, params, some):
    rates = ExchangeRates(row['start_day'], locale_en=True)
    usd_rate = float(rates['USD'].value)
    if usd_rate >= 0:
        price = float(row['price'])
        row['usd'] = round(usd_rate, 2)
        row['price_usd'] = round(price / usd_rate, 2)
    else:
        row['price_usd'] = 0
        row['usd'] = 0
    return row


def defineUSDCourese(data_1, data_2):
    data_1 = data_1.apply(UseCourseDate, args=([], 1), axis=1)
    data_2 = data_2.apply(UseCourseDate, args=([], 1), axis=1)
    return data_1, data_2


# TODO УНИФИКАЦИЯ тренировчных наборов
def normolizeTrainsData(train, train_new):
    """
    Унификация датасетов - те же поля и тот же тип данныъх по столбцам
    train -  изначачальный датасет
    train_new -  спарсенный датасет с сайта

    :param train:
    :param train_new:
    :return train:
    """
    # предобработка столцов
    train.columns = [col.replace('\r', '').replace('\n', '').strip() for col in train.columns]
    train_new.columns = [col.replace('\r', '').replace('\n', '').strip() for col in train_new.columns]

    train.dropna(subset=['bodyType', 'modelDate', 'numberOfDoors', 'enginePower', 'engineDisplacement',
                         'price'], inplace=True)
    train_new.dropna(subset=['bodyType', 'modelDate', 'numberOfDoors', 'enginePower', 'engineDisplacement',
                             'price'], inplace=True)

    columnc = []
    #  нет поля car_url
    train['car_url'] = ''
    columnc.append('car_url')

    # bodyType -
    pd.options.display.max_rows = 1000
    train, train_new = Column_lower_strip(train, train_new, 'bodyType')
    train, train_new = replaceBodyType(train, train_new, 'bodyType')
    columnc.append('bodyType')

    # brand
    train, train_new = Column_lower_strip(train, train_new, 'brand')
    columnc.append('brand')

    # color
    train = reDefineColor(train, 'color')
    train, train_new = Column_lower_strip(train, train_new, 'color')
    columnc.append('color')

    # fuelType
    train, train_new = Column_lower_strip(train, train_new, 'fuelType')
    columnc.append('fuelType')

    # modelDate
    train, train_new = preColumnToInt(train, train_new, 'modelDate')
    columnc.append('modelDate')

    # numberOfDoors
    train, train_new = preColumnToInt(train, train_new, 'numberOfDoors')
    columnc.append('numberOfDoors')

    # productionDate
    train, train_new = preColumnToInt(train, train_new, 'productionDate')
    columnc.append('productionDate')

    # vehicleTransmission
    train = trainVehicleTransmission(train, 'vehicleTransmission')
    train, train_new = Column_lower_strip(train, train_new, 'vehicleTransmission')
    columnc.append('vehicleTransmission')

    #  engineDisplacement - объем дигателя
    train = trainEnginEDisplacement(train, 'engineDisplacement')
    train, train_new = preColumnToFloat(train, train_new, 'engineDisplacement')
    train['engineDisplacement'].fillna(0, inplace=True)
    train_new['engineDisplacement'].fillna(0, inplace=True)
    columnc.append('engineDisplacement')

    #  enginePower - объем дигателя - удалим отсуствующие строки
    train.dropna(subset=['enginePower'], inplace=True)
    train_new.dropna(subset=['enginePower'], inplace=True)
    columnc.append('enginePower')

    # description - описание
    train['description'].fillna('', inplace=True)
    train_new['description'].fillna('', inplace=True)
    columnc.append('description')

    # mileage - пробег (по умолчанию )
    train['mileage'].fillna(0, inplace=True)
    train_new['mileage'].fillna(0, inplace=True)
    columnc.append('mileage')

    # drivertrain (Привод) - удалим отсуствующие значения
    train.rename({'Привод': 'drivertrain'}, axis=1, inplace=True)
    train.dropna(subset=['drivertrain'], inplace=True)

    train_new.rename({'Привод': 'drivertrain'}, axis=1, inplace=True)
    train_new.dropna(subset=['drivertrain'], inplace=True)
    train, train_new = Column_lower_strip(train, train_new, 'drivertrain')
    columnc.append('drivertrain')

    # driverSide (Руль) - удалим отсуствующие значения
    train.rename({'Руль': 'driverSide'}, axis=1, inplace=True)
    train.dropna(subset=['driverSide'], inplace=True)

    train_new.rename({'Руль': 'driverSide'}, axis=1, inplace=True)
    train_new.dropna(subset=['driverSide'], inplace=True)
    columnc.append('driverSide')

    # ownersCount (Владельцы) - владельцы количество ( по умолчанию 0)
    #  представим это значение как категориальное
    train.rename({'Владельцы': 'ownersCount'}, axis=1, inplace=True)
    train['ownersCount'].fillna(0, inplace=True)

    train_new.rename({'Владельцы': 'ownersCount'}, axis=1, inplace=True)
    train_new['ownersCount'].fillna(0, inplace=True)
    train, train_new = trainOnwerCounts(train, train_new, 'ownersCount')
    columnc.append('ownersCount')

    # pts (ПТС) - ПТС
    train.rename({'ПТС': 'pts'}, axis=1, inplace=True)
    train_new.rename({'ПТС': 'pts'}, axis=1, inplace=True)

    train['pts'].fillna('ORIGINAL', inplace=True)
    train_new['pts'].fillna('ORIGINAL', inplace=True)
    columnc.append('pts')

    #  state - Состояние
    train.rename({'Состояние': 'state'}, axis=1, inplace=True)
    train_new.rename({'Состояние': 'state'}, axis=1, inplace=True)
    train['state'].fillna(repair_state_no, inplace=True)
    train_new['state'].fillna(repair_state_no, inplace=True)
    train, train_new = Column_lower_strip(train, train_new, 'state')
    train, train_new = trainDefineState(train, train_new, 'state')
    columnc.append('state')

    # customs  (Таможня) -  Таможня
    train.rename({'Таможня': 'customs'}, axis=1, inplace=True)
    train_new.rename({'Таможня': 'customs'}, axis=1, inplace=True)
    columnc.append('customs')

    # start_date  - удлаим пропуски + выделим из даты год (для вычисления новх признаков)
    train = redefineDateToInix(train, 'start_date')
    train_new = reDefineTimstampToDate(train_new, True, 'start_date')
    columnc.append('start_year')
    columnc.append('start_day')

    # model - модель авто
    train_new = getSellIdModel_fromURL(train_new)
    train, train_new = Column_lower_strip(train, train_new, 'model')
    columnc.append('model')

    # vendor - Страна
    train, train_new = defineVendor(train, train_new, 'brand')
    columnc.append('vendor')

    # racing - время разгона
    train, train_new = defineRacing(train, train_new, 'racing')
    columnc.append('racing')

    # price_usd - цена в долларах
    # train, train_new = defineUSDCourese(train, train_new)
    train['price_log'] = np.log(train['price'])
    train_new['price_log'] = np.log(train_new['price'])
    columnc.append('price')
    columnc.append('price_log')
    # columnc.append('price_usd')
    # columnc.append('usd')

    # sell_id
    train['sell_id'] = 0
    train_new['sell_id'] = 0
    columnc.append('sell_id')

    train = train[columnc]
    train_new = train_new[columnc]

    bigdata = train.append(train_new, ignore_index=True)
    return bigdata


#  TODO Унификация тестовых данных
def normolizeTestData(test):
    test.columns = [col.replace('\r', '').replace('\n', '').strip() for col in test.columns]

    columnc = []
    columnc.append('car_url')

    # bodyType -
    test = Column_lower_strip(test, None, 'bodyType')
    columnc.append('bodyType')

    # brand
    test = Column_lower_strip(test, None, 'brand')
    columnc.append('brand')

    # color
    test = Column_lower_strip(test, None, 'color')
    columnc.append('color')

    # fuelType
    test = Column_lower_strip(test, None, 'fuelType')
    columnc.append('fuelType')

    # modelDate
    test = preColumnToInt(test, None, 'modelDate')
    columnc.append('modelDate')

    # numberOfDoors
    test = preColumnToInt(test, None, 'numberOfDoors')
    columnc.append('numberOfDoors')

    # productionDate
    test = preColumnToInt(test, None, 'productionDate')
    columnc.append('productionDate')

    # vehicleTransmission
    test = testTransmission(test)
    test = Column_lower_strip(test, None, 'vehicleTransmission')
    columnc.append('vehicleTransmission')

    #  engineDisplacement - объем дигателя (Для неизвестного значения примем -1)
    test['engineDisplacement'] = test['engineDisplacement'].str.strip()
    test['engineDisplacement'] = test['engineDisplacement'].astype(str).str.replace('LTR', '', regex=False)
    test['engineDisplacement'] = test['engineDisplacement'].replace(r'', '-1', regex=False)
    test['engineDisplacement'] = test['engineDisplacement'].astype(float)
    columnc.append('engineDisplacement')

    #  enginePower - объем дигателя - удалим отсуствующие строки
    test['enginePower'] = test['enginePower'].str.replace('N12', '', regex=False)
    test['enginePower'] = test['enginePower'].astype(int)
    columnc.append('enginePower')

    # description - описание
    test['description'].fillna('', inplace=True)
    columnc.append('description')

    # mileage - пробег (по умолчанию )
    test['mileage'].fillna(0, inplace=True)
    columnc.append('mileage')

    # drivertrain (Привод) - удалим отсуствующие значения
    # test.rename({'Привод': 'drivertrain'}, axis=1, inplace=True)
    test['drivertrain'] = test['Привод']
    columnc.append('drivertrain')

    # driverSide (Руль) - удалим отсуствующие значения
    test.rename({'Руль': 'driverSide'}, axis=1, inplace=True)
    test = Column_lower_strip(test, None, 'driverSide')
    test = testDriveSide(test, 'driverSide')
    columnc.append('driverSide')

    # ownersCount (Владельцы) - владельцы количество ( по умолчанию 0)
    #  представим это значение как категориальное
    test.rename({'Владельцы': 'ownersCount'}, axis=1, inplace=True)
    test = Column_lower_strip(test, None, 'ownersCount')
    test = testOnwerCounts(test, 'ownersCount')
    columnc.append('ownersCount')

    # pts (ПТС) -  (если нет инфы то  ORIGINAL)
    test.rename({'ПТС': 'pts'}, axis=1, inplace=True)
    test = Column_lower_strip(test, None, 'pts')
    test = testPTS(test, 'pts')
    test['pts'].fillna('ORIGINAL', inplace=True)
    columnc.append('pts')

    # state - Состояние
    test.rename({'Состояние': 'state'}, axis=1, inplace=True)
    test = Column_lower_strip(test, None, 'state')
    test = trainDefineState(test, None, 'state')
    columnc.append('state')

    # customs  (Таможня) -  Таможня
    test.rename({'Таможня': 'customs'}, axis=1, inplace=True)
    test = Column_lower_strip(test, None, 'customs')
    test = testСustoms(test, 'customs')
    columnc.append('customs')

    # start_date  - удлаим пропуски + выделим из даты год (для вычисления новх признаков)
    test = reDefineTimstampToDate(test, False, 'parsing_unixtime')
    columnc.append('start_year')
    columnc.append('start_day')

    # model - модель авто
    test = Column_lower_strip(test, None, 'model_name')
    test['model'] = test['model_name']
    test = Column_lower_strip(test, None, 'model')
    columnc.append('model')

    # vendor - Страна
    test['vendor'] = test['vendor'].str.strip()
    columnc.append('vendor')

    # racing - время разгона
    test['racing'] = test['racing'].astype(float)
    columnc.append('racing')

    # price_usd - цена в долларах
    # train, train_new = defineUSDCourese(train, train_new)
    test['price'] = 0
    test['price_log'] = 0
    test['price_log'] = 0
    columnc.append('price')
    columnc.append('price_log')
    # columnc.append('price_usd')
    # columnc.append('usd')

    # sell_id
    columnc.append('sell_id')
    test = test[columnc]
    return test


RANDOM_SEED = 42
VERSION = 16
DIR_TRAIN = 'input/'
DIR_TEST = 'input/'
VAL_SIZE = 0.20  # 20%

# TODO Чтение первоначальных признаков
train_new = pd.read_csv(DIR_TRAIN + 'Base.csv', lineterminator='\n')
train = pd.read_csv(DIR_TRAIN + 'all_auto_ru_09_09_2020.csv', lineterminator='\n')
#
test = pd.read_csv(DIR_TEST + 'test.csv')
sample_submission = pd.read_csv(DIR_TEST + 'sample_submission.csv')

#
# # # TODO Унификация признаков
# train = normolizeTrainsData(train, train_new)
# train.to_csv(DIR_TRAIN + 'Base_all.csv', index=False, encoding="utf-8-sig")


#
test_normize = normolizeTestData(test)
test_normize.to_csv(DIR_TRAIN + 'test_all.csv', index=False, encoding="utf-8-sig")



# TODo объединение
train_formated = pd.read_csv(DIR_TEST + 'Base_all.csv')
test_formated = pd.read_csv(DIR_TEST + 'test_all.csv')

train_formated['sample'] = 1
test_formated['sample'] = 0
data = test_formated.append(train_formated, sort=False).reset_index(drop=True)

# TODO NEW ПРИЗНАКИ
data['production_model'] = data['productionDate'] - data['modelDate']

data['car_age'] = data['start_year'] - data['productionDate']
data['car_age'] = data['car_age'].apply(lambda x: x if x > 0 else 0)

data['descrip_lenth'] = data['description'].apply(lambda x: len(str(x)))

data['mileage_year'] = pd.to_numeric(data['mileage'] / data['car_age']).round(2)
data['mileage_year'].replace([np.inf, -np.inf], np.nan, inplace=True)
data['mileage_year'].fillna(0, inplace=True)
data['mileage_year'] = data['mileage_year'].astype(int)

# TODO mileage - пробег и установим бинарный признак новая/неновое авто
CAR_NEW_LIMIT = 500


def defineMileageRow(row, params, some):
    if int(row['mileage']) <= CAR_NEW_LIMIT:
        row['car_new'] = 1
    else:
        row['car_new'] = 0
    return row


def defineMileageType(data):
    data = data.apply(defineMileageRow, args=([], 1), axis=1)
    return data


data = defineMileageType(data)

num_cols = ['modelDate', 'productionDate', 'production_model', 'enginePower', 'engineDisplacement', 'car_age',
            'mileage', 'racing', 'descrip_lenth', 'mileage_year']
bin_cols = ['condition', 'customs', 'driveSide', 'transmission', 'tcp']
cat_cols = ['brand', 'bodyType', 'color', 'fuelType', 'vehicleTransmission', 'drivertrain', 'vendor', 'ownersCount',
            'numberOfDoors', 'vehicleTransmission']

num_cols.remove('modelDate')

print(imp_num)
exit()


# TODo ANALIZ
def analis_number_scatterplot(date, col):
    fig, axes = plt.subplots(figsize=(30, 8))

    sns.scatterplot(x=col, y="price", data=date)
    plt.xticks(rotation=45)
    plt.xlabel(col, fontsize=20)
    plt.title('распределенеи по годам - ' + col, fontsize=20)
    plt.show()


def analis_number_boxplot(data, col):
    fig, axes = plt.subplots(figsize=(30, 8))
    sns.boxplot(data[col])
    plt.xticks(rotation=45)
    plt.xlabel(col, fontsize=20)
    plt.title('boxplot - ' + col, fontsize=20)
    plt.show()


def analiz_number_feature(data, col, sample=1):
    if sample == 0 or sample == 1:
        train = data.query('sample == ' + str(sample))
    else:
        train = data
    fig, axes = plt.subplots(figsize=(30, 8))

    sns.distplot(train[col])
    plt.xticks(rotation=45)
    plt.xlabel(col, fontsize=20)
    plt.ylabel('Кол.', rotation='horizontal', fontsize=20)
    plt.title('Распределение  числового признака - ' + col, fontsize=20)
    plt.show()
    analis_number_boxplot(train, col)
    analis_number_scatterplot(train, col)


bin_cols = ['condition', 'customs', 'driveSide', 'transmission', 'tcp']

# ПЕРВОНЧАЛЬНОЕ ЗНАЧЕНИЕ
train.dropna(subset=['productionDate', 'mileage', 'car_class'], inplace=True)
train.dropna(subset=['price'], inplace=True)

columns = ['bodyType', 'brand', 'productionDate', 'engineDisplacement', 'mileage']
df_train = train[columns]
df_test = test[columns]
y = train['price']

df_train['sample'] = 1
df_test['sample'] = 0

data = df_test.append(df_train, sort=False).reset_index(drop=True)
for colum in ['bodyType', 'brand', 'engineDisplacement']:
    data[colum] = data[colum].astype('category').cat.codes

X = data.query('sample == 1').drop(['sample'], axis=1)
X_sub = data.query('sample == 0').drop(['sample'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)

rf = xgb.XGBRegressor(random_state = RANDOM_SEED)

model = CatBoostRegressor(iterations=5000, random_seed=RANDOM_SEED,eval_metric='MAPE',ustom_metric=['R2', 'MAE'],
                          silent=True,)
model.fit(X_train, y_train,
          # cat_features=cat_features_ids,
          eval_set=(X_test, y_test),
          verbose_eval=0,
          use_best_model=True,
          # plot=True
          )
predict = model.predict(X_test)
print(f"Точность модели по метрике MAPE: {(mape(y_test, predict)) * 100:0.2f}%")

