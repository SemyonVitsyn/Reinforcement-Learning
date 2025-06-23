from collections import defaultdict
import random
import math
import numpy as np


class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Агент, настраиваемый с помощью Q-обучения
        основано на https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Поля объекта класса
          - self.epsilon (вероятность совершения случайного действия)
          - self.alpha (шаг обучения)
          - self.discount (параметр дисконтирования gamma)

        Рекомендуемые к использованию методы класса
          - self.get_legal_actions(state) {состояние state, хэшируемое -> список действий, каждое хэшируемое}
            возвращает возможные действия для состояния state
          - self.get_qvalue(state,action)
            возвращает Q(state,action)
          - self.set_qvalue(state,action,value)
            присваивает Q(state,action) := value
        !!!Внимание!!!
        Замечание: просьба не использовать self._qValues напрямую. 
            Для этого есть методы self.get_qvalue/set_qvalue.
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Возвращает Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Присваивает значение Q-функции ценности для [state,action] равное value """
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Вычисляет для агента оценку V(s) с помощью текущих значений Q-функции ценности
        V(s) = max_over_action Q(state,action) максимум берётся по возможным действиям.
        Замечание: значения Q-функции ценности могут быть произвольного знака.
        """
        possible_actions = self.get_legal_actions(state)

        # Если нет доступных действий, возвращаем 0.
        if len(possible_actions) == 0:
            return 0.

        value = max([self.get_qvalue(state, a) for a in possible_actions])
        return value

    def update(self, state, action, reward, next_state, done):
        """
        Метод для обновления значения Q-функции ценности:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # параметры агента
        gamma = self.discount
        learning_rate = self.alpha
        
        q = reward + gamma * (1 - done) * self.get_value(next_state)
        q = (1 - learning_rate) * self.get_qvalue(state, action) + learning_rate * q

        self.set_qvalue(state, action, q)

    def get_best_action(self, state):
        """
        Поиск наилучшего действия для состояния state по текущим значениям Q-функции ценности. 
        """
        possible_actions = self.get_legal_actions(state)

        # Если нет доступных действий, возвращаемы None
        if len(possible_actions) == 0:
            return None

        idx = np.argmax([self.get_qvalue(state, a) for a in possible_actions])

        return possible_actions[idx]

    def get_action(self, state):
        """
        Вычисление действия, совершаемого в текущем состоянии state с учётом исследования среды.
        С вероятностью self.epsilon действие сэмплируется из равномерного распределения.
        В противоположном случае --- жадно выбирается наилучшее действие (self.get_best_action).

        Замечание: воспользуйтесь random.choice(list) для равномерного сэмплирования из списка list.
              Для сэмплированя из множества {"True", "False"} с заданной вероятность рекомендуется
              сгенерировать равномерную случайную величину из отрезка [0, 1] и сравнить с заданной
              вероятностью: если больше --- возвращаем один сэмпл, иначе --- другой сэмпл.
        """

        # Выбор действия
        possible_actions = self.get_legal_actions(state)
        action = None

        # Если нет доступных действий, возвращаемы None
        if len(possible_actions) == 0:
            return None

        # параметры агента
        epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.choice(possible_actions)
        
        return self.get_best_action(state)

