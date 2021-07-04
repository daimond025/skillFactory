import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn
import statistics


class PrepareDate():
    # в констуркоре определим методы обработки данных
    def __init__(self, df_output):
        self.df_output = df_output

    # расчет затрат
    # (количество топлива (http://newsruss.ru/doc/index.php/Расход_топлива_у_самолётов) * минуты  )
    # * стоимость топлива (https://favt.gov.ru/dejatelnost-ajeroporty-i-ajerodromy-ceny-na-aviagsm/?id=7379) - 51 859
    #  для самолтатов  Boeing 737-300 - 40 кг/1 мин, Sukhoi Superjet-100 - 28 кг / 1 мин
    def calculatePetrol(self):
        self.df_output['petrol'] = 0
        self.df_output.loc[(self.df_output['plane_model'] == 'Boeing 737-300'), 'petrol'] = 40
        self.df_output.loc[(self.df_output['plane_model'] == 'Sukhoi Superjet-100'), 'petrol'] = 28

        self.df_output['cost_petrol'] = (self.df_output['petrol'] * self.df_output['duration_minutes']) / 1000 * 51859
        self.calcululateTotalCosts()

    # так как общих затрат я не нашел - буду исходить из процентого соотношения затрат (https://www.aex.ru/docs/1/2011/4/13/1323/)
    #  топливо - 20% (оно нам известно  - примем за константу)
    #  аэропортовые сборы, наземное обслуживание, аэронавигация  примем на 20% (равно типливу)
    #  лизинговые и таможенные платежи, расходы на обслуживание кредита примем на 20% (равно типливу)
    #  техническое обслуживание парка авиасудов примем на 10% (равно половину типливу)
    #  расходы на оплату труда пилотного состава на 10% (равно половину типливу)
    #  todo общиие затраты =  4 * (затраты топлива)
    def calcululateTotalCosts(self):
        self.df_output['Costs'] = 4 * self.df_output['cost_petrol']

    #  извлечение даты
    def foramedDate(self):
        self.df_output['date_departure'] = pd.to_datetime( self.df_output['date_departure'], format='%d-%m-%Y %H:%M:%S')
        self.df_output['day_of_week'] =  self.df_output['date_departure'].dt.dayofweek
        self.df_output['month'] =  self.df_output['date_departure'].dt.month

    #  извлечение разных метрик
    def calculateProfit(self):
        # общая прибыль = продажа билетов - затраы
        self.df_output['profit'] =  self.df_output['pass_summ'] - self.df_output['Costs']
        #  заполняемость мест в т.ч. бизнес и экном классов
        self.df_output['occupancy'] = self.df_output['pass_count'] / self.df_output['plane_count']
        self.df_output['occupancy_econom'] = self.df_output['pass_econom_count'] / self.df_output['plane_econom_count']
        self.df_output['occupancy_bisness'] = self.df_output['pass_bisiness_count'] / self.df_output['plane_bisiness_count']

         # доля продажи  бизнес и экном билетов от общей продажи билетов
        self.df_output['part_bisness'] = self.df_output['pass_bisiness_summ'] / self.df_output['pass_summ']
        self.df_output['part_econom'] = self.df_output['pass_econom_summ'] / self.df_output['pass_summ']

        # как один пассажир влияет на доходность / или на продажу билетов
        self.df_output['deposit_profit'] = (self.df_output['profit'] )/ self.df_output['pass_count']
        self.df_output['deposit'] = (self.df_output['pass_summ'] )/ self.df_output['pass_count']
        self.df_output['deposit_econom'] = (self.df_output['pass_econom_summ'] )/ self.df_output['pass_econom_count']
        self.df_output['deposit_bisness'] = (self.df_output['pass_bisiness_summ'] )/ self.df_output['pass_bisiness_count']


    def processData(self):
        self.calculatePetrol()
        self.foramedDate()
        self.calculateProfit()

        return self.df_output


data = pd.read_csv('data.csv')

df_object = PrepareDate(data)
df = df_object.processData()
df = df[df['city'] != 'Novokuznetsk'].copy()


seaborn.barplot(x="plane_model", y="city", hue="city", data=df)
# seaborn.barplot(x=df['city'].value_counts().index, y=df['city'].value_counts())
