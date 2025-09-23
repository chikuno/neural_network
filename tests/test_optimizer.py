import torch
from train.optimizer import MetaOptimizer
from model.mlp import AdvancedMLP


def test_metaoptimizer_api():
    model = AdvancedMLP(input_size=10, hidden_layers=[8], output_size=5)
    opt = MetaOptimizer(model, base_lr=0.01)
    # Ensure zero_grad exists and does not error
    opt.zero_grad()
    # Create a dummy loss and call step(loss)
    x = torch.randn(2, 10)
    out = model(x)
    # if multi-task returns list, pick first
    if isinstance(out, list):
        out = out[0]
    target = torch.zeros(out.size(0), dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(out, target)
    opt.step(loss)
    # calling step without loss should also work (assumes caller did backward)
    opt.zero_grad()
    out2 = model(x)
    if isinstance(out2, list):
        out2 = out2[0]
    loss2 = loss_fn(out2, target)
    loss2.backward()
    opt.step(None)
