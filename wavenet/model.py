import os
import torch as t

from wavenet.networks import WaveNet as WaveNetModule

class Wavenet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr=0.002):
        self.net = WaveNetModule(layer_size, stack_size, in_channels, res_channels)

        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        self._prepare_for_gpu()

    @staticmethod
    def _loss():
        loss = t.nn.CrossEntropyLoss()

        if t.cuda.is_available():
            loss.cuda()

        return loss

    def _optimizer(self):
        optimizer = t.optim.Adam(self.net.parameters(), lr=self.lr)

        if t.cuda.is_available():
            optimizer.cuda()

        return optimizer

    def _prepare_for_gpu(self):
        if t.cuda.device_count() > 1:
            print(f'{t.cuda.device_count()} GPUs detected')
            self.net = t.nn.DataParallel(self.net)

        if t.cuda.is_available():
            self.net.cuda()

    def train(self, inputs, targets):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, timestep, channels]
        :return: float loss
        """
        outputs = self.net(inputs)

        loss = self.loss(outputs.view(-1, self.in_channels), targets.long().view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def generate(self, inputs):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        return self.net(inputs)

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, f'{basename}_{step}.pkl')
        else:
            return os.path.join(model_dir, f'{basename}.pkl')

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print(f"Loading model from {model_dir}")

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(t.load(model_path))

    def save(self, model_dir, step=0):
        print(f"Saving model into {model_dir}")

        model_path = self.get_model_path(model_dir, step)

        t.save(self.net.state_dict(), model_path)   