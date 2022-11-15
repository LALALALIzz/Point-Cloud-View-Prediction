import torch

if __name__ == '__main__':
    x = torch.tensor([[[1, 2]],
                      [[3, 4]],
                      [[5, 6]]])
    y = torch.tensor([1, 2, 3])
    y = y.repeat(x.shape[0], 1, 1)
    print(torch.concat((y,x), dim=-1).shape)
    print(y.shape)
    print(x.shape)