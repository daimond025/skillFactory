import numpy as np

"Границы нахождения искмого значения"
border_down = 1
border_up = 100


def game_core_my(number):
    """
    Алгоритм делит рассматриваемый диапазон  пополам и смотрит в каком промежутке находится искомое число
    """

    """количество попыток 
    отгадывания"""
    count = 1

    """Границы поиска числа"""
    mark_down = border_down
    mark_upper = border_up

    predict = 0

    while number != predict:
        count += 1

        # разбиение на диапазон
        centre = int((mark_down + mark_upper + 1) / 2)

        # сдвиг границ
        if number >= centre:
            mark_down = centre
        else:
            mark_upper = centre

        # сравнение  определенных границ с искомым числом
        if mark_down == number:
            predict = mark_down
        elif mark_upper == number:
            predict = mark_upper

    return count


def score_game(game_core):
    """Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число"""
    count_ls = []
    np.random.seed(1)
    random_array = np.random.randint(border_down, border_up + 1, size=1000)
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return score


score_game(game_core_my)
