from torchviz import make_dot, make_dot_from_trace
import torch
import torch.nn.functional as F
import torch.optim as optim


def save_dot(model, name):
    with torch.onnx.set_training(model, False):
        x = torch.randn(1, 33)
        y = torch.randn(1, 4)
        trace, _ = torch.jit.get_trace_graph(model, args=(x,y))
        make_dot_from_trace(trace).save(name, '.')
