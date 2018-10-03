from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYERS = [48, 24, 12]


class AbstractQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layer_specs):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(AbstractQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        if not layer_specs:
            layer_specs = LAYERS

        layers = OrderedDict()
        layers['l0'] = nn.Linear(state_size, layer_specs[0])
        layers['r0'] = nn.ReLU()

        for i in range(len(layer_specs) - 1):
            layers['l' + str(i + 1)] = nn.Linear(layer_specs[i], layer_specs[i + 1])
            layers['r' + str(i + 1)] = nn.ReLU()
        layers['ll'] = nn.Linear(layer_specs[-1], action_size)
        self.layers = layers


class QNetwork(AbstractQNetwork):
    def __init__(self, state_size, action_size, seed, layer_specs):
        super(QNetwork, self).__init__(state_size, action_size, seed, layer_specs)
        self.seq = nn.Sequential(self.layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.seq.forward(state)


class DuelingQNetwork(QNetwork):
    def __init__(self, state_size, action_size, seed, layer_specs):
        super(DuelingQNetwork, self).__init__(state_size, action_size, seed, layer_specs)
        self.adv = nn.Sequential(self.layers)
        self.val = nn.Sequential(self.layers)
        # self.adv1 = nn.Linear(action_size, action_size * 2)
        # self.adv2 = nn.Linear(action_size * 2, action_size)
        #
        # self.val1 = nn.Linear(action_size, action_size * 2)
        # self.val2 = nn.Linear(action_size * 2, 1)

    def forward(self, state):
        val = self.val.forward(state)
        adv = self.adv.forward(state)

        # f = super(DuelingQNetwork, self).forward(state)
        # f = F.relu(f)
        # adv = F.relu(self.adv1(f))
        # adv = self.adv2(adv)
        #
        # val = F.relu(self.val1(f))
        # val = self.val2(val)

        return val + adv - adv.mean()


class ConvQNetwork(nn.Module):

    def __init__(self):
        super(ConvQNetwork, self).__init__()

        self.conv_a1 = nn.Conv2d(12, 12, kernel_size=(4,4), padding=(2,2))
        self.bn_a1 = nn.BatchNorm2d(12)
        self.conv_a2 = nn.Conv2d(12, 24, kernel_size=(4,4), padding=(2,2))
        self.bn_a2 = nn.BatchNorm2d(24)
        self.conv_a3 = nn.Conv2d(24, 48, kernel_size=(4,4), padding=(2,2))
        self.bn_a3 = nn.BatchNorm2d(48)
        self.conv_a4 = nn.Conv2d(48, 96, kernel_size=(4,4), padding=(2,2))
        self.bn_a4 = nn.BatchNorm2d(96)


        # self.conv_a1 = nn.Conv3d(3, 12, kernel_size=(1, 4, 4),  stride=(1, 2, 2))
        # self.conv_a2 = nn.Conv3d(12, 24, kernel_size=(1, 4, 4),  stride=(1, 2, 2))
        # self.conv_a3 = nn.Conv3d(24, 48, kernel_size=(3, 4, 4),  stride=(1, 2, 2))

        self.conv_b1 = nn.Conv2d(12, 24, kernel_size=(3, 3), stride=(3, 3))
        self.bn_b1 = nn.BatchNorm2d(24)
        self.conv_b2 = nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(3, 3))
        self.bn_b2 = nn.BatchNorm2d(48)
        self.conv_b3 = nn.Conv2d(48, 96, kernel_size=(3, 3), stride=(3, 3))
        self.bn_b3 = nn.BatchNorm2d(96)
        self.conv_b4 = nn.Conv2d(96, 24, kernel_size=3)
        self.bn_b4 = nn.BatchNorm2d(24)


        self.conv_c1 = nn.Conv2d(12, 24, kernel_size=(7,2), stride=2)
        self.bn_c1 = nn.BatchNorm2d(24)
        self.conv_c2 = nn.Conv2d(24, 48, kernel_size=(7,2), stride=2)
        self.bn_c2 = nn.BatchNorm2d(48)
        self.conv_c3 = nn.Conv2d(48, 96, kernel_size=(7,2), stride=2)
        self.bn_c3 = nn.BatchNorm2d(96)
        self.conv_c4 = nn.Conv2d(96, 24, kernel_size=4)
        self.bn_c4 = nn.BatchNorm2d(24)

        self.conv_d1 = nn.Conv2d(12, 24, kernel_size=(2, 7), stride=2)
        self.bn_d1 = nn.BatchNorm2d(24)
        self.conv_d2 = nn.Conv2d(24, 48, kernel_size=(2, 7), stride=2)
        self.bn_d2 = nn.BatchNorm2d(48)
        self.conv_d3 = nn.Conv2d(48, 96, kernel_size=(2, 7), stride=2)
        self.bn_d3 = nn.BatchNorm2d(96)
        self.conv_d4 = nn.Conv2d(96, 24,kernel_size=4)
        self.bn_d4 = nn.BatchNorm2d(24)

        # 256
        #
        #1152
        #1728
        self.head1 = nn.Linear(1896, 1024)
        self.head2 = nn.Linear(1024, 24)

        #self.out = nn.Linear(24, 4)

        self.adv1 = nn.Linear(24, 12)
        self.adv2 = nn.Linear(12, 4)

        self.val1 = nn.Linear(24, 12)
        self.val2 = nn.Linear(12, 1)

        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 nn.init.xavier_uniform_(m.weight)
                 #nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, nn.Conv3d):
                 nn.init.xavier_uniform_(m.weight)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)

        a = F.relu(F.max_pool2d(self.bn_a1(self.conv_a1(x)), (4,4)))
        a = F.relu(F.max_pool2d(self.bn_a2(self.conv_a2(a)), (2,2)))
        a = F.relu(F.max_pool2d(self.bn_a3(self.conv_a3(a)), (2,2)))
        a = F.relu(F.max_pool2d(self.bn_a4(self.conv_a4(a)), (2,2)))
        a = a.view(a.size(0), -1)

        # a = F.relu(self.conv_a1(x))
        # a = F.relu(F.max_pool3d(self.conv_a2(a),(1,2,2)))
        # a = F.relu(self.conv_a3(a))
        # a = a.view(a.size(0), -1)

        b = F.relu(self.bn_b1(self.conv_b1(x)))
        b = F.relu(self.bn_b2(self.conv_b2(b)))
        b = F.relu(self.bn_b3(self.conv_b3(b)))
        b = F.relu(self.bn_b4(self.conv_b4(b)))
        b = b.view(b.size(0), -1)

        c = F.relu(self.bn_c1(self.conv_c1(x)))
        c = F.relu(self.bn_c2(self.conv_c2(c)))
        c = F.relu(self.bn_c3(self.conv_c3(c)))
        c = F.relu(self.bn_c4(self.conv_c4(c)))
        c = c.view(c.size(0), -1)

        d = F.relu(self.bn_d1(self.conv_d1(x)))
        d = F.relu(self.bn_d2(self.conv_d2(d)))
        d = F.relu(self.bn_d3(self.conv_d3(d)))
        d = F.relu(self.bn_d4(self.conv_d4(d)))
        d = d.view(d.size(0), -1)

        x = torch.cat((a,b,c,d), 1)
        #x = b
        x = F.relu(self.head1(x))
        x = F.relu(self.head2(x))

        #return self.out(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
