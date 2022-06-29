import torch as t
import torch.nn as nn

from wavenet.exceptions import InputSizeError

class CausalConv1d(nn.Module):
    '''Causal Convolutional for WaveNet'''

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()

        self.padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding, # "same"
            bias=False,
        )

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, t.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)
        if self.padding != 0:
            return output[:, :, :-self.padding]
        return output

class DilatedCausalConv1d(nn.Module):
    '''Dilated Causal Convolution for WaveNet'''

    def __init__(self, channels, dilation, kernel_size=2):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=1,
            padding=self.padding, # "same"
            dilation=dilation,
            bias=False,
        )

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, t.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)
        if self.padding != 0:
            return output[:, :, :-self.padding]
        return output


class ResidualBlock(nn.Module):
    '''Residual Block for WaveNet'''

    def __init__(self, res_channels, skip_channels, dilation):
        '''
        
        Parameters
        ----------
        res_channels : int
            Number of residual channels for input, output.
        skip_channels : int
            Number of skip channels for output.
        dilation : int
            Dilation rate for dilated convolution.            
        '''
        super().__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()

    def forward(self, x, skip_size):
        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual connection
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return output, skip

class ResidualStack(nn.Module):
    '''Stacks of layers of Residual Block for WaveNet'''

    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        '''
        
        Parameters
        ----------
        layer_size : int
            Number of layers in the residual block.
            10 = layer[dilation=1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        stack_size : int
            Number of residual blocks in the stack.
            5 = stack[layer1, layer2, layer3, layer4, layer5]
        res_channels : int
            Number of residual channels for input, output.
        skip_channels : int
            Number of skip channels for output.
        '''
        super().__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if t.cuda.device_count() > 1:
            block = t.nn.DataParallel(block)

        if t.cuda.is_available():
            block.cuda()

        return block

    def build_dilations(self):
        dilations = []

        for s in range(self.stack_size):
            for l in range(self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, skip_channels):
        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            # res_block = self._residual_block(res_channels, skip_channels, dilation)
            res_block = ResidualBlock(res_channels, skip_channels, dilation)
            res_blocks.append(res_block)

        return res_blocks

    def forward(self, x):
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            output, skip = res_block(output)
            skip_connections.append(skip)
        
        return t.stack(skip_connections)

class DenseNet(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv1d(channels, channels, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)

        output = self.softmax(output)

        return output

class WaveNet(nn.Module):
    
    def __init__(self, layer_size, stack_size, in_channels, res_channels):
        '''
        
        Parameters
        ----------
        layer_size : int
            Number of layers in the residual block.
        stack_size : int
            Number of residual blocks in the stack.
        in_channels : int
            Number of channels for input data, same as skip channels.
        res_channels : int
            Number of residual channels for input, output.
        '''
        super().__init__()

        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)

        self.causal = CausalConv1d(in_channels, res_channels)

        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels)

        self.dense_net = DenseNet(in_channels)

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = sum(layers)

        return int(num_receptive_fields)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields

        self.check_input_size(x, output_size)

        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)

    def forward(self, x):
        '''
        The size of timestep(3rd dimension) has to bigger than receptive fields

        Parameters
        ----------
        x : torch.Tensor[batch_size, time_steps, channels]
            Input data.
        '''
        output = x.transpose(1, 2)

        output_size = self.calc_output_size(output)

        output = self.causal(output)

        skip_connections = self.res_stack(output, output_size)

        output = t.sum(skip_connections, dim=0)

        output = self.dense_net(output)

        return output.transpose(1, 2).contiguous()