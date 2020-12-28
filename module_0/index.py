import numpy as np

"Границы нахождения искмого значения"
border_down = 1
border_up = 100


def game_core_my(number):
    """Алгоритм делит рассматриваемый диапазон  пополам и смотрит в каком промежутке искомое число нахожится
    Кроме этого если при процессе разбиения одна граница повторяется значит искомое число находится где-то с
    этой границей. Воодится уточняющий коэффициент step который итерационно уменьшается, но он еще изменяет изменяет
    границы и устанавливает их более точно"""

    count = 1

    """Границы поиска числа"""
    mark_down = border_down
    mark_upper = border_up

    """Границы поиска числа, 
    дублируют границы для индексации повтора границ"""
    mark_down_temp = mark_down
    mark_upper_temp = mark_upper

    # шаг уточнения границ
    step = 20

    predict = 0
    while number != predict:
        count += 1

        # разбиение на диапазон
        centre = int((mark_down + mark_upper) / 2)

        # сдвиг границ
        if number >= centre:
            mark_down = centre
            mark_down_temp = centre
        else:
            mark_upper = centre
            mark_upper_temp = centre

        # уточнение границ с учетом шага step
        if mark_down == mark_down_temp:
            if number >= mark_down + step:
                mark_down += step
            if number <= mark_upper - step:
                mark_upper -= step
            step = int(step/2)
        elif mark_upper == mark_upper_temp:
            if number >= mark_down + step:
                mark_down += step
            if number <= mark_upper - step:
                mark_upper -= step
            step = int(step/2)

        # сравнение границ с искомым числом
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
