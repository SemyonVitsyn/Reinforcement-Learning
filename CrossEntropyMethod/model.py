import torch
import torch.nn as nn
import torch.optim.optimizer
from abc import ABC, abstractmethod


class MLPBaseModel(nn.Module):

    def __init__(
            self,
            input_dim: int = 6,
            output_dim: int = 3,
            inner_dims: list = [128],
            activation_func=nn.Tanh()):
        """
        Базовая модель многослойного перцептрона (MLP).

        Args:
            input_dim (int, optional): Размерность входного состояния. По умолчанию 6.
            output_dim (int, optional): Размерность выходного действия. По умолчанию 3.
            inner_dims (list, optional): Список с количеством нейронов в каждом скрытом слое. По умолчанию [12, 36, 64, 64, 32, 6].
            activation_func (optional): Функция активации. По умолчанию nn.ReLU().
        """
        super().__init__()

        ##########################
        # нужно описать инициализацию модели с линейными слоями и функцией активации activation_func
        # первый слой должен принимать на вход input_dim параметров, а последний слой должен возвращать output_dim
        # все другие слои должны содержать количество нейронов, соответственно списку inner_dims
        ##########################

        layerlist = []

        layerlist.append(nn.Linear(input_dim, inner_dims[0]))
        layerlist.append(activation_func)

        for i in range(1, len(inner_dims)):
            layerlist.append(nn.Linear(inner_dims[i-1], inner_dims[i]))
            layerlist.append(activation_func)

        layerlist.append(nn.Linear(inner_dims[-1], output_dim))

        self.layers = nn.Sequential(*layerlist)

    def _reset_params(self):
        """
        Сбрасывает параметры модели.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):
        """
        Прямой проход через модель.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, input_dim).

        Return:
            torch.Tensor: Выходной тензор формы (batch_size, output_dim).
        """
        return self.layers(x)


class MLPBaseWrapper(ABC, nn.Module):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr: float,
                 loss: callable,
                 mean: torch.Tensor = None,
                 std: torch.Tensor = None):
        """
        Базовый класс-обертка для модели MLP.

        Args:
            model (nn.Module): Модель MLP.
            optimizer (torch.optim.Optimizer): Оптимизатор.
            lr (float): Скорость обучения.
            loss (callable): Функция потерь.
            mean (torch.Tensor, optional): Средние значения для нормализации. По умолчанию None.
            std (torch.Tensor, optional): Стандартные отклонения для нормализации. По умолчанию None.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr)
        self.loss_fn = loss
        self.mean = mean
        self.std = std

    def forward(self, x):
        """
        Прямой проход через модель.

        Args:
            x (torch.Tensor): Входной тензор.

        Return:
            torch.Tensor: Выходной тензор.
        """
        if self.mean is not None and self.std is not None:
            return self.model((x - self.mean)/self.std)
        return self.model(x)

    def _reset_params(self):
        """
        Сбрасывает параметры модели.
        """
        self.model._reset_params()

    def set_learning_rate(self, new_learning_rate: float = 1e-3) -> None:
        """
        Устанавливает новую скорость обучения.

        Args:
            new_learning_rate (float, optional): Новая скорость обучения. По умолчанию 1e-3.
        """
        self.lr = new_learning_rate
        self.optimizer = self.optimizer.__class__(
            self.model.parameters(), lr=self.lr)

    def set_optimizer(
            self,
            new_optimizer: torch.optim.Optimizer = torch.optim.AdamW) -> None:
        """
        Устанавливает новый оптимизатор.

        Args:
            new_optimizer (torch.optim.Optimizer, optional): Новый оптимизатор. По умолчанию torch.optim.AdamW.
        """
        self.optimizer = new_optimizer(self.model.parameters(), self.lr)

    def set_normalization_params(self, env_observation_spase):
        low = None
        high = None
        try:
            low = env_observation_spase.low
            high = env_observation_spase.high
        except Exception as e:
            print(e, 'most likely, you passed an incorrect parameter to the class method. The input should be evn.observation_space')
            return
        self.mean = (high + low)/2
        self.std = (high - low)/2

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError(
            'this method is abstract, please override it')

    @abstractmethod
    def partial_fit(self, inputs, targets):
        raise NotImplementedError(
            'this method is abstract, please override it')

    @abstractmethod
    def get_task(self):
        raise NotImplementedError(
            'this method is abstract, please override it')


class MLPRegressorModel(MLPBaseWrapper):
    def __init__(
            self,
            model=MLPBaseModel(),
            optimizer=torch.optim.AdamW,
            lr=0.001,
            loss=torch.nn.SmoothL1Loss()):
        """
        Модель MLP для задачи регрессии.
        """
        super().__init__(model, optimizer, lr, loss)

    def predict(self, x):
        """
        Предсказание для задачи регрессии.

        Args:
            x: Входные данные.

        Return:
            torch.Tensor: Предсказанные значения.
        """
        with torch.no_grad():
            return self.model(x)

    def partial_fit(self, inputs, targets):
        """
        Частичное обучение для задачи регрессии.

        Args:
            inputs: Входные данные в модель.
            targets: Целевые значения.

        Return:
            float: Значение функции потерь.
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        ##########################
        # нужно реализовать шаг обучения модели с оптимизатором и learning rate'ом, заданным в классе
        # для этого нужно получить предсказание модели на inputs, после получить лосс и прокинуть его (loss.backward()) и сделать шаг оптимайзера
        ##########################

        preds = self.predict(inputs)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @property
    def get_task(self):
        """
        Возвращает тип задачи.

        Return:
            str: "Regression".
        """
        return 'Regression'


class MLPClassifireModel(MLPBaseWrapper):
    def __init__(
            self,
            model=MLPBaseModel(),
            optimizer=torch.optim.AdamW,
            lr=0.001,
            loss=torch.nn.CrossEntropyLoss()):
        """
        Модель MLP для задачи классификации.
        """
        super().__init__(model, optimizer, lr, loss)

    def predict(self, x):
        """
        Предсказание для задачи классификации.

        Args:
            x: Входные данные.

        Return:
            torch.Tensor: Вероятности классов.
        """
        with torch.no_grad():
            logits: torch.Tensor = self.model(x)
            if (len(logits.shape) == 1):
                probabilities = torch.softmax(logits, dim=0)
            else:
                probabilities = torch.softmax(logits, dim=1)
            return probabilities

    def partial_fit(self, inputs, targets):
        """
        Частичное обучение для задачи классификации.

        Args:
            inputs: Входные данные в модель.
            targets: Целевые значения.

        Return:
            float: Значение функции потерь.
        """
        input = torch.tensor(inputs, dtype=torch.float32)
        target = torch.tensor(targets, dtype=torch.long)

        ##########################
        # нужно реализовать шаг обучения модели с оптимизатором и learning rate'ом, заданным в классе
        # для этого нужно получить предсказание модели на inputs, после получить лосс и прокинуть его (loss.backward()) и сделать шаг оптимайзера
        ##########################

        self.optimizer.zero_grad()
        
        logits: torch.Tensor = self.model(input)
        if (len(logits.shape) == 1):
            pred = torch.softmax(logits, dim=0)
        else:
            pred = torch.softmax(logits, dim=1)

        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @property
    def get_task(self):
        """
        Возвращает тип задачи.

        Return:
            str: "Classification".
        """
        return 'Classification'
