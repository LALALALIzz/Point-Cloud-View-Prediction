import torch

def hidden_state(h, x):
    return h ** 2 + 3 * x

if __name__ == '__main__':
    x0 = torch.tensor(1.0, requires_grad=True)
    x1 = torch.tensor(2.0, requires_grad=True)
    x = x0.detach()
    xx = x1.detach()

    h0 = torch.tensor(2.0, requires_grad=True)

    h1 = h0 ** 2 + x ** 2

    h2 = h1 ** 2 + x ** 2
    h2.backward()
    print(x.grad)
