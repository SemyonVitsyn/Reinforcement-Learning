from typing import Union, Callable
from dataclasses import dataclass

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import errno
import warnings

from gymnasium.wrappers import RecordVideo
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed, parallel_config
from IPython.display import clear_output
from collections import deque
from model import MLPClassifireModel, MLPRegressorModel


def show_progress(rewards_batch: torch.Tensor, log: list,
                  percentile: int, reward_range=[-1050, +10]):
    """
    Визуализирует процесс обучения, отображая графики наград и их перцентили.

    Args:
        rewards_batch (torch.Tensor): Тензор с наградами за сессии.
        log (list): Лог данных для построения графиков.
        percentile (int): Значение перцентиля для расчета порога.
        reward_range (list, optional): Диапазон значений для гистограммы наград. По умолчанию [-1050, +10].

    Return:
        None
    """

    mean_reward = np.array(rewards_batch).mean() # средняя награда
    threshold = np.percentile(np.array(rewards_batch), percentile) # порог награды

    print("средняя награда = %.3f, порог = %.3f" % (mean_reward, threshold))
    log.append((mean_reward, threshold))

    ##########################
    # график с историей порогов наград и средних вознаграждений в зависимости от номера итеркции обучения
    ##########################
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot([item[0] for item in log], label="Средняя награда")
    plt.plot([item[1] for item in log], label=f"{percentile}% перцетиль")
    plt.xlabel("Iter")
    plt.ylabel("Reward")
    plt.legend()

    ##########################
    # гистограма с распределением вознаграждений за сессию и уровнем выбранного перцентиля
    ##########################
    plt.subplot(1, 2, 2)
    plt.hist(np.array(rewards_batch), range=reward_range)
    plt.axvline(threshold, color="red")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

    clear_output(True)
    plt.show()


@dataclass
class session_params():
    """
    Параметры для настройки сессии.

    Args:
        max_step (int, optional): Максимальное количество шагов в сессии. По умолчанию 1000.
        test (bool, optional): Режим тестирования. По умолчанию False.
        epsilon (float, optional): Параметр для регуляризации в режиме регрессии. По умолчанию 1e-2.
    """
    max_step: int = 1000
    test: bool = False
    epsilon: float = 1e-2


@dataclass
class train_agent_params():
    """
    Параметры для обучения агента.

    Args:
        train_steps (int, optional): Количество шагов обучения. По умолчанию 100.
        session_quantity (int, optional): Количество сессий на шаг обучения. По умолчанию 100.
        percentile (int, optional): Перцентиль для отбора лучших сессий. По умолчанию 70.
        goal_score (int, optional): Целевое значение награды для завершения обучения. По умолчанию 150.
        history_length (int, optional): Длина истории для хранения и использования сессий. По умолчанию 1.
        verbose (bool, optional): Режим вывода информации. По умолчанию False.
        parallel (bool, optional): Использование параллельных вычислений. По умолчанию False.
        n_workers (int, optional): Количество рабочих процессов. По умолчанию 1.
        strict (bool, optional): Строгий режим отбора сессий. По умолчанию False.
    """
    train_steps: int = 100
    session_quantity: int = 100
    percentile: int = 70
    goal_score: int = 150
    history_length: int = 1
    verbose: bool = False
    parallel: bool = False
    n_workers: int = 1
    strict: bool = False


class GYMAgentWrapper():
    @staticmethod
    def select_best_session(
            states_batch,
            actions_batch,
            rewards_batch,
            percentile: int = 70,
            strict: bool = False):
        """
        Отбирает лучшие сессии на основе заданного перцентиля.

        Args:
            states_batch: Список состояний из всех сессий.
            actions_batch: Список действий из всех сессий.
            rewards_batch: Список наград из всех сессий.
            percentile (int, optional): Перцентиль для отбора. По умолчанию 70.
            strict (bool, optional): Строгий режим отбора. По умолчанию False.

        Return:
            tuple: (elite_states, elite_actions) - отобранные состояния и действия.
        """
        reward_threshold = np.percentile(rewards_batch, percentile) # порог наград, расчитывается с помощью перцентиля

        elite_states = []
        elite_actions = []

        ##########################
        # отбор states и actions, при которых награда больше заданного перцентиля.
        # при strict == True, отобранны должны быть только states и actions с вознограждением > перцентиля
        # при strict == False, отобранны должны быть только states и actions с вознограждением >= перцентиля
        ##########################

        if strict:
            for states, actions, reward in zip(states_batch, actions_batch, rewards_batch):
                if reward > reward_threshold:
                    elite_states.extend(states)
                    elite_actions.extend(actions)
        else:
            for states, actions, reward in zip(states_batch, actions_batch, rewards_batch):
                if reward >= reward_threshold:
                    elite_states.extend(states)
                    elite_actions.extend(actions)

        return elite_states, elite_actions

    @staticmethod
    def generate_session_wrapper(agent, session_params, env_name):
        env = gym.make(env_name, agent.session_params.max_step)
        res = agent._generate_session(
            other_env=env, session_params=session_params)
        del env
        return res

    def __init__(self,
                 gym_env: gym.Env,
                 agent: Union[MLPClassifireModel,
                              MLPRegressorModel],
                 show_progress_func: Callable = show_progress):
        """
        Инициализация обертки для агента и среды.

        Args:
            gym_env (gym.Env): Среда Gymnasium.
            agent (Union[MLPClassifireModel, MLPRegressorModel]): Агент для обучения.
            show_progress_func (Callable, optional): Функция для отображения прогресса. По умолчанию show_progress.
        """
        self.gym_env = gym_env
        self.agent = agent
        self.agent_task = self.agent.get_task
        self.session_params = session_params()
        self.train_agent_params = train_agent_params()
        self.show_progress_func = show_progress_func

    @property
    def __get_env_space_info(self):
        """
        Возвращает информацию о пространстве действий и состояний среды.

        Return:
            dict: Словарь с ключами 'action_space' и 'observation_space'.
        """
        return {'action_space': self.gym_env.action_space,
                'observation_space': self.gym_env.observation_space}

    def _generate_session(self, other_env: gym.Env = None, session_params: session_params = None, *args, **kwargs):
        """
        Генерирует одну сессию взаимодействия агента со средой.

        Args:
            other_env (optional): Альтернативная среда для генерации сессии. По умолчанию None.
            session_params (optional): Альтернативные параметры для инициализации сесиии. По умолчанию None
            *args, **kwargs: Дополнительные аргументы для reset среды.

        Return:
            tuple: (states, actions, total_reward) - состояния, действия и общая награда за сессию.
        """
        if session_params == None:
            session_params = self.session_params

        states, actions = [], []
        total_reward = 0
        env = None
        if other_env == None:
            env = self.gym_env
        else:
            env = other_env
        s, _ = env.reset()
        n_actions = 0
        if self.agent_task == 'Classification':
            n_actions = self.__get_env_space_info['action_space'].n
        elif self.agent_task == 'Regression':
            n_actions = sum(self.__get_env_space_info['action_space'].shape)
        else:
            raise NotImplementedError(
                f"task {self.agent_task} don't supported")
        for _ in range(session_params.max_step):
            if self.agent_task == 'Classification':
                agent_action = self.agent.predict(
                    torch.tensor(np.array([s])).reshape(-1))
                agent_action = np.array(agent_action)

                assert agent_action.shape == (
                    n_actions,), "Нужно получить вектор вероятностей"

                if session_params.test:
                    a = np.argmax(agent_action)
                else:
                    a = np.random.choice(
                        np.arange(n_actions), p=agent_action)
            elif self.agent_task == 'Regression':
                agent_action = self.agent.predict(torch.tensor(np.array([s])))

                if session_params.test:
                    a = agent_action[0]
                else:
                    a = np.random.normal(agent_action.detach(
                    ).numpy(), session_params.epsilon)[0]

            else:
                raise NotImplementedError(
                    f"task {self.agent_task} don't supported")

            new_s, r, terminated, truncated, _ = env.step(a)

            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s
            if terminated or truncated:
                break

        del s, new_s, a

        return states, actions, total_reward

    def reset_agent(self, *args, **kwargs):
        """
        Сбрасывает параметры агента.

        Args:
            *args, **kwargs: Дополнительные аргументы для сброса.
        """
        self.agent._reset_params()

    def reset_gym_env(self, *args, **kwargs):
        """
        Сбрасывает среду Gymnasium.

        Args:
            *args, **kwargs: Дополнительные аргументы для сброса.
        """
        self.gym_env.reset()

    def reset_all_params(self, *args, **kwargs):
        """
        Сбрасывает все параметры агента, среды и настроек.
        """
        self.agent._reset_params()
        self.gym_env.reset()
        self.session_params = session_params()
        self.train_agent_params = train_agent_params()

    def set_sesion_params(
            self,
            max_step: int = 1000,
            test: bool = False,
            epsilon: float = 1e-2) -> None:
        """
        Устанавливает параметры сессии.

        Args:
            max_step (int, optional): Максимальное количество шагов. По умолчанию 1000.
            test (bool, optional): Режим тестирования. По умолчанию False.
            epsilon (float, optional): Параметр для регуляризации. По умолчанию 1e-2.
        """
        del self.session_params
        self.session_params = session_params(
            max_step=max_step, test=test, epsilon=epsilon)

    def set_train_agent_params(
            self,
            train_steps: int = 100,
            session_quantity: int = 100,
            percentile: int = 70,
            goal_score: int = 150,
            history_length: int = 1,
            verbose: bool = False,
            parallel: bool = False,
            n_workers: int = 1,
            strict: bool = False) -> None:
        """
        Устанавливает параметры обучения агента.

        Args:
            train_steps (int, optional): Количество шагов обучения. По умолчанию 100.
            session_quantity (int, optional): Количество сессий на шаг. По умолчанию 100.
            percentile (int, optional): Перцентиль для отбора. По умолчанию 70.
            goal_score (int, optional): Целевая награда. По умолчанию 150.
            history_length (int, optional): Длина истории для хранения и использования сессий. По умолчанию 1.
            verbose (bool, optional): Режим вывода информации. По умолчанию False.
            parallel (bool, optional): Использование параллельных вычислений. По умолчанию False.
            n_workers (int, optional): Количество рабочих процессов. По умолчанию 1.
            strict (bool, optional): Строгий режим отбора. По умолчанию False.
        """
        del self.train_agent_params
        self.train_agent_params = train_agent_params(
            train_steps=train_steps,
            session_quantity=session_quantity,
            percentile=percentile,
            goal_score=goal_score,
            history_length=history_length,
            verbose=verbose,
            parallel=parallel,
            n_workers=n_workers,
            strict=strict)

    def train_agent(self):
        """
        Обучает агента на основе заданных параметров.

        Return:
            list: Лог данных с информацией о наградах и порогах.
        """
        if self.train_agent_params.verbose:
            print(f"Шаг агента = {self.agent.lr}", flush=True)

        log = []
        sessions = deque([], self.train_agent_params.history_length *
                         self.train_agent_params.session_quantity)

        for i in range(self.train_agent_params.train_steps):
            if self.train_agent_params.n_workers == 1:
                sessions.extend([self._generate_session() for _ in range(
                    self.train_agent_params.session_quantity)])
            # elif self.train_agent_params.n_workers > 1:
            #     n_workers = self.train_agent_params.n_workers
            #     with parallel_config(backend="loky", prefer="threads", inner_max_num_threads=1):
            #         ##########################
            #         # нужно релизовать параллельную генерацию сессий
            #         # смотри joblib.Parallel и joblib.delayed
            #         # настоятельно рекомендуем использовать generate_session_wrapper staticmethod для возможности сереализации процесса, иначе велика вероятность что код не запустится
            #         ##########################
            #     sessions.extend(
            #         # your code here
            #     )
            else:
                raise ValueError("quantity of workers can't be less that 1")

            states_batch, actions_batch, rewards_batch = zip(*sessions)
            elite_states, elite_actions = self.select_best_session(
                states_batch, actions_batch, rewards_batch, self.train_agent_params.percentile, self.train_agent_params.strict)

            if self.agent_task == 'Classification':
                self.agent.partial_fit(elite_states, elite_actions)
            elif self.agent_task == 'Regression':
                # if (len(elite_actions.shape) >= 2) and (
                #         elite_actions.shape[1] == 1):
                #     elite_actions = elite_actions.reshape(-1)
                self.agent.partial_fit(elite_states, elite_actions)
            else:
                raise NotImplementedError(
                    f"task {self.agent_task} don't supported")

            # log data
            log.append([np.mean(rewards_batch), np.percentile(rewards_batch,
                                                              self.train_agent_params.percentile)])

            if self.train_agent_params.verbose:
                inter_min = np.min(rewards_batch)
                min_lim = -self.train_agent_params.train_steps if - \
                    self.train_agent_params.train_steps < inter_min else inter_min
                inter_max = np.max(rewards_batch)
                max_lim = self.train_agent_params.goal_score if self.train_agent_params.goal_score > inter_max else inter_max
                self.show_progress_func(
                    rewards_batch,
                    log,
                    self.train_agent_params.percentile,
                    reward_range=[
                        min_lim,
                        max_lim])

                if np.mean(
                        rewards_batch) > self.train_agent_params.goal_score:
                    mean_reward = np.mean(rewards_batch)
                    threshold = np.percentile(
                        rewards_batch, self.train_agent_params.percentile)
                    print("средняя награда = %.3f, порог=%.3f" %
                          (mean_reward, threshold))
                    print(
                        "Вы выиграли! Можете прервать процедуру обучения с помощью сигнала KeyboardInterrupt.")
                    return log

            if self.train_agent_params.verbose:
                mean_reward = np.mean(rewards_batch)
                threshold = np.percentile(
                    rewards_batch, self.train_agent_params.percentile)
                print("средняя награда = %.3f, порог=%.3f" %
                      (mean_reward, threshold))
                del mean_reward, threshold

            del states_batch, actions_batch, rewards_batch, elite_states, elite_actions

        del sessions

        return log

    def record_wideo(self,
                     video_folder_path: Union[str,
                                              Path],
                     video_quantity: int):
        """
        Записывает видео с взаимодействием агента и среды.

        Args:
            video_folder_path (Union[str, Path]): Путь к папке для сохранения видео.
            video_quantity (int): Количество видео для записи.
        """
        video_folder_path = Path(video_folder_path)
        if os.path.exists(video_folder_path) == False:
            warnings.warn(
                f"folder {video_folder_path} don't exist. The creation process has started.")
            try:
                os.makedirs(video_folder_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if os.path.isdir(video_folder_path) == False:
            raise ValueError(
                f"folder {video_folder_path} don't folder")

        with RecordVideo(
            env=self.gym_env,
            video_folder=video_folder_path,
            episode_trigger=lambda episode_number: True,
        ) as env_monitor:
            sessions = [self._generate_session(env_monitor)
                        for _ in range(video_quantity)]
