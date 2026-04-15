import torch


def symlog(x):
    """Symmetric log: sign(x) * ln(|x| + 1). Compresses large reward scales."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def bottle(f, x_tuple):
    """Reshape [time, batch, ...] -> [time*batch, ...], apply f, reshape back."""
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


def cal_returns(reward, value, bootstrap, pcont, lambda_):
    """
    Compute lambda-returns for imagined trajectories (Dreamer eq. 5-6).

    Args:
        reward, value: imagined rewards/values  [horizon, batch, 1]
        bootstrap:     last predicted value      [batch, 1]
        pcont:         discount factor           scalar or [horizon, batch, 1]
        lambda_:       lambda for TD(lambda)

    Returns:
        target returns  [horizon, batch, 1]
    """
    assert list(reward.shape) == list(value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)

    next_value = torch.cat((value[1:], bootstrap[None]), 0)
    inputs = reward + pcont * next_value * (1 - lambda_)
    outputs = []
    last = bootstrap
    for t in reversed(range(reward.shape[0])):
        last = inputs[t] + pcont[t] * lambda_ * last
        outputs.append(last)
    return torch.flip(torch.stack(outputs), [0])
