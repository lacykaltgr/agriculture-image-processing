import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.nn import init
import numpy as np


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9,
                 hiddens=None, dropouts=None, kernel_sizes=None, maxpools=None, paddings=None, strides=None,
                 criterion=nn.CrossEntropyLoss(), activation=nn.ReLU(), output_activation=nn.Softmax(dim=1),
                 pre_process=None, post_process=None, dimensions=2, device='cuda'):
        super(AttentionUNet, self).__init__()

        if dimensions == 2:
            conv = nn.Conv2d
            batchnorm = nn.BatchNorm2d
            dropout = nn.Dropout2d
            maxpool = nn.MaxPool2d
        elif dimensions == 3:
            conv = nn.Conv3d
            batchnorm = nn.BatchNorm3d
            dropout = nn.Dropout3d
            maxpool = nn.MaxPool3d
        else:
            raise ValueError('Only 2D and 3D convolutions are supported')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.criterion = criterion
        self.pre_process = pre_process if pre_process is not None else lambda x: x
        self.post_process = post_process if post_process is not None else lambda x: x
        self.device = device

        # channels
        if hiddens is None:
            hiddens = [64]
        elif isinstance(hiddens, int):
            hiddens = [hiddens]
        elif isinstance(hiddens, list):
            assert len(hiddens) > 0, "Number of hiddens must be greater than 0"
        else:
            raise ValueError("Hiddens must be an integer or a list of integers")
        channels_downsample = [in_channels] + hiddens
        self.n_downsamples = len(channels_downsample) - 1
        channels_upsample = hiddens[::-1] + [out_channels]
        self.n_upsamples = len(channels_upsample) - 2

        # dropouts
        if dropouts is None:
            dropouts = [0] * self.n_downsamples
        elif isinstance(dropouts, float):
            dropouts = [dropouts] * self.n_downsamples
        elif isinstance(dropouts, list):
            assert len(dropouts) == self.n_downsamples, \
                "Number of dropouts must match the number of downsamples"
        else:
            raise ValueError("Dropouts must be a float or a list of floats")

        # maxpools
        if maxpools is None:
            maxpools = [2] * self.n_downsamples
        elif isinstance(maxpools, int):
            maxpools = [maxpools] * self.n_downsamples
        elif isinstance(maxpools, tuple):
            maxpools = [maxpools] * self.n_downsamples
        elif isinstance(maxpools, list):
            assert len(maxpools) == self.n_downsamples, \
                "Number of maxpools must match the number of downsamples"
        else:
            raise ValueError("Maxpools must be an integer, a tuple of integers or a list of integers")

        n_enc_convolutions = 2 * self.n_downsamples
        n_dec_convolutions = 3 * self.n_upsamples + 2
        # kernel_sizes
        if kernel_sizes is None:
            kernel_sizes = [[3] * n_enc_convolutions, [3] * n_dec_convolutions]
        elif isinstance(kernel_sizes, int):
            kernel_sizes = [[kernel_sizes] * n_enc_convolutions, [kernel_sizes] * n_dec_convolutions]
        elif isinstance(kernel_sizes, tuple):
            kernel_sizes = [[kernel_sizes] * n_enc_convolutions, [kernel_sizes] * n_dec_convolutions]
        elif isinstance(kernel_sizes, list):
            assert len(kernel_sizes) == 2, \
                "Number of kernel_sizes must be organized into encoder, decoder kernel sizes"
            assert len(kernel_sizes[0]) == n_enc_convolutions, \
                "Number of encoder kernel_sizes must match the number of encoder convolutions"
            assert len(kernel_sizes[1]) == n_dec_convolutions, \
                "Number of decoder kernel_sizes must match the number of decoder convolutions"
        else:
            raise ValueError("Kernel_sizes must be an integer, a tuple of integers or a list of integers")

        enc_kernel_sizes, dec_kernel_sizes = kernel_sizes

        if paddings is None:
            paddings = [['same'] * n_enc_convolutions, ['same'] * n_dec_convolutions]
        elif isinstance(paddings, str):
            paddings = [[paddings] * n_enc_convolutions, [paddings] * n_dec_convolutions]
        elif isinstance(paddings, tuple):
            paddings = [[paddings] * n_enc_convolutions, [paddings] * n_dec_convolutions]
        elif isinstance(paddings, int):
            paddings = [[paddings] * n_enc_convolutions, [paddings] * n_dec_convolutions]
        elif isinstance(paddings, list):
            assert len(paddings) == 2, \
                "Number of paddings must be organized into encoder, decoder paddings"
            assert len(paddings[0]) == n_enc_convolutions, \
                "Number of encoder paddings must match the number of encoder convolutions"
            assert len(paddings[1]) == n_dec_convolutions, \
                "Number of decoder paddings must match the number of decoder convolutions"
        else:
            raise ValueError("Paddings must be an integer, a tuple of integers or a list of integers")
        enc_paddings, dec_paddings = paddings

        if strides is None:
            strides = [[1] * n_enc_convolutions, [1] * n_dec_convolutions]
        elif isinstance(strides, int):
            strides = [[strides] * n_enc_convolutions, [strides] * n_dec_convolutions]
        elif isinstance(strides, tuple):
            strides = [[strides] * n_enc_convolutions, [strides] * n_dec_convolutions]
        elif isinstance(strides, list):
            assert len(strides) == 2, \
                "Number of strides must be organized into encoder, decoder strides"
            assert len(strides[0]) == n_enc_convolutions, \
                "Number of encoder strides must match the number of encoder convolutions"
            assert len(strides[1]) == n_dec_convolutions, \
                "Number of decoder strides must match the number of decoder convolutions"
        else:
            raise ValueError("Strides must be an integer, a tuple of integers or a list of integers")
        enc_strides, dec_strides = strides

        self.layers_upsample = nn.ModuleDict()
        self.layers_downsample = nn.ModuleDict()
        self.attention = nn.ModuleDict()

        # Left side of the U-Net
        for i in range(self.n_downsamples):
            self.layers_downsample[f"conv_enc_{i}_1"] = \
                conv(channels_downsample[i], channels_downsample[i + 1],
                     kernel_size=enc_kernel_sizes[2 * i], padding=enc_paddings[2 * i], stride=enc_strides[2 * i])

            self.layers_downsample[f"conv_enc_{i}_2"] = \
                conv(channels_downsample[i + 1], channels_downsample[i + 1],
                     kernel_size=enc_kernel_sizes[2 * i + 1], padding=enc_paddings[2 * i + 1],
                     stride=enc_strides[2 * i + 1])

            self.layers_downsample[f"batchnorm_enc_{i}"] = \
                batchnorm(channels_downsample[i + 1])

            self.layers_downsample[f"dropout_{i}"] = \
                dropout(p=dropouts[i])

            if i != self.n_downsamples - 1:  # utsonak nem kell
                self.layers_downsample[f"maxpool_{i}"] = maxpool(maxpools[i])

        self.gating = UnetGridGatingSignal3(
            channels_downsample[-1], channels_upsample[0], kernel_size=(1, 1), is_batchnorm=True,
        )

        # Up-sampling starts, right side of the U-Net
        for i in range(self.n_upsamples):

            self.attention[f"attention_{i}"] = \
                MultiAttentionBlock(
                    channels_upsample[i+1], channels_upsample[i], channels_upsample[i+1],
                    nonlocal_mode='concatenation', sub_sample_factor=(2, 2)
                )

            self.layers_upsample[f"upconv_dec_{i}"] = \
                conv(channels_upsample[i], channels_upsample[i + 1],
                     kernel_size=dec_kernel_sizes[3 * i], padding=dec_paddings[3 * i], stride=dec_strides[3 * i])

            self.layers_upsample[f"conv_dec_{i}_1"] = \
                conv(channels_upsample[i], channels_upsample[i + 1],
                     kernel_size=dec_kernel_sizes[3 * i + 1], padding=dec_paddings[3 * i + 1],
                     stride=dec_strides[3 * i + 1])

            self.layers_upsample[f"conv_dec_{i}_2"] = \
                conv(channels_upsample[i + 1], channels_upsample[i + 1],
                     kernel_size=dec_kernel_sizes[3 * i + 2], padding=dec_paddings[3 * i + 2],
                     stride=dec_strides[3 * i + 2])

            if i == self.n_upsamples - 1:
                self.layers_upsample[f"conv_dec_{i}_3"] = \
                    conv(channels_upsample[i + 1], channels_upsample[i + 1],
                         kernel_size=dec_kernel_sizes[3 * i + 3], padding=dec_paddings[3 * i + 3],
                         stride=dec_strides[3 * i + 3])
            self.layers_upsample[f"batchnorm_dec_{i}"] = batchnorm(channels_upsample[i + 1])
            # else:
            #    self.layers_upsample[f"batchnorm_dec_{i}"] = batchnorm(channels_upsample[i + 1])

        # Output layer of the U-Net with a softmax activation
        self.layers_upsample["conv_out"] = \
            conv(channels_upsample[-2], channels_upsample[-1],
                 kernel_size=dec_kernel_sizes[-1], padding=dec_paddings[-1], stride=dec_strides[-1])
        self.output_activation = output_activation
        self.activation = activation

        # initialize all parameters with xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        skip_connections = []
        # Left side of the U-Net
        for i in range(self.n_downsamples):
            x = self.activation(self.layers_downsample[f"conv_enc_{i}_1"](x))
            x = self.activation(self.layers_downsample[f"conv_enc_{i}_2"](x))
            x = self.layers_downsample[f"batchnorm_enc_{i}"](x)
            x = self.layers_downsample[f"dropout_{i}"](x)

            if i != self.n_downsamples - 1:
                skip_connections.append(x)
                x = self.layers_downsample[f"maxpool_{i}"](x)

        gating = self.gating(x)

        # Upsampling Starts, right side of the U-Net
        for i in range(self.n_upsamples):
            skip = skip_connections.pop()
            gate, _ = self.attention[f"attention_{i}"](skip, gating)

            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = self.activation(self.layers_upsample[f"upconv_dec_{i}"](x))

            x = torch.cat([gate, x], dim=1)

            x = self.activation(self.layers_upsample[f"conv_dec_{i}_1"](x))
            x = self.activation(self.layers_upsample[f"conv_dec_{i}_2"](x))

            gating = x

            if i == self.n_upsamples - 1:
                x = self.activation(self.layers_upsample[f"conv_dec_{i}_3"](x))
            x = self.layers_upsample[f"batchnorm_dec_{i}"](x)

        # Output layer of the U-Net with a softmax activation
        out_conv = self.layers_upsample["conv_out"](x)
        y = self.output_activation(out_conv)
        return y

    def train_model(self, train_loader, valid_loader, early_stopper,
                    num_epochs=100, learning_rate=1e-4, weight_decay=1e-5, device='cuda'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loss = valid_loss = []
        for epoch in range(num_epochs):
            epoch_train_loss = correct_train = total_train = 0
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

                c, t = self._accuracy_score(targets, outputs)
                correct_train += c
                total_train += t

                outputs.detach().cpu()
                targets.detach().cpu()

            train_accuracy = 100 * correct_train / total_train

            epoch_valid_loss, valid_accuracy = self.evaluate(valid_loader, device=device)

            epoch_train_loss /= len(train_loader)
            epoch_valid_loss /= len(valid_loader)
            train_loss.append(epoch_train_loss)
            valid_loss.append(epoch_valid_loss)
            print(
                f'Epoch {epoch + 1:03}: | Train Loss: {epoch_train_loss:.5f} | Validation Loss: {epoch_valid_loss:.5f}'
                + f' | Train Acc: {train_accuracy:.2f}% | Valid Acc: {valid_accuracy:.2f}%')

            if early_stopper.early_stop(epoch_valid_loss):
                break
        return train_loss, valid_loss

    def predict(self, test_loader, device='cuda'):
        self.eval()
        raw_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, _ = batch
                inputs = self.pre_process(inputs).to(device)
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                raw_predictions.append(preds.detach().cpu().numpy())
        results = []
        for batch in raw_predictions:
            for img in batch:
                results.append(self.post_process(img))
        return results, raw_predictions

    def evaluate(self, test_loader, device='cuda'):
        epoch_valid_loss = 0
        self.eval()
        correct_valid = 0
        total_valid = 0
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            epoch_valid_loss += loss.item()

            c, t = self._accuracy_score(targets, outputs)
            correct_valid += c
            total_valid += t

            outputs.detach().cpu()
            targets.detach().cpu()

        valid_accuracy = 100 * correct_valid / total_valid
        return epoch_valid_loss, valid_accuracy

    def summary(self, input_shape):
        from torchinfo import summary
        return summary(self, input_size=input_shape)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        if y_true.shape[1] == 1:
            total = np.prod(y_true.size())
            correct = torch.sum(y_pred.round() == y_true).item()
            return correct, total
        else:
            _, acc_predictions = torch.max(y_pred, 1)
            _, acc_targets = torch.max(y_true, 1)
            total = np.prod(acc_targets.size())
            correct = (acc_predictions == acc_targets).sum().item()
            return correct, total


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor,
                             padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1,
                           kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            weights_init_kaiming(m)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        """
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        """

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1: continue
            weights_init_kaiming(m)

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)