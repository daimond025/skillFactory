from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


def run():
    driver = webdriver.Chrome(executable_path='chromedriver.exe')
    driver.get("https://tproger.ru/quiz/real-programmer/")

    button = driver.find_element_by_class_name("quiz_button")
    button.click()

    begin_index = 255
    while begin_index < 267:
        item_id = f"quiz_item_{begin_index}"

        # тест вопроса
        question = driver.find_element_by_xpath('//*[@id="' + str(item_id) + '"]/div[1]').text

        # нажимаем на кнопку выбора ответа
        answer = driver.find_element_by_xpath('//*[@id="' + str(item_id) + '"]/div[2]/div[1]')
        answer.click()

        # выбор привильного ответа
        right_ans = driver.find_element_by_xpath('//*[@id="' + str(item_id) + '"]/div[2]/div[1]').text

        print('Вопрос  -  ', question)
        print('Правильный ответа  -  ', right_ans)
        print('-------------------------------------------')

        # на след вопрос
        button_nxt = driver.find_element_by_xpath('//*[@id="' + str(item_id) + '"]/button')
        button_nxt.click()

        begin_index += 1

run()

