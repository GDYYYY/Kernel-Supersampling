import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if checkpoint.ok:
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    loader = data.Data(args)                ##data loader

    # loader test
    # print(loader.loader_test)

    # print("dataset len:",len(loader.loader_test))
    # idx = 0
    # import numpy as np
    # import imageio
    # for data in loader.loader_test:
    #     lr, hr, name = data
    #     print(lr.shape)
    #     print(hr.shape)
    #     print(name)

    #     lr = lr[0,...].numpy() * 255.
    #     lr = lr.transpose([1, 2, 0]).astype(np.uint8)
    #     imageio.imwrite('Set1.jpg', lr)

    #     break

    # loader traindata
    # print(loader.loader_train)

    # print("dataset len:",len(loader.loader_train))
    # idx = 0
    # import numpy as np
    # import imageio
    # for batch, (lr, hr, name) in enumerate(loader.loader_train):
    #     print(batch)
    #     print(lr.shape)
    #     print(hr.shape)
    #     print(name)

    #     lr = lr[0,...].numpy() * 255.
    #     lr = lr.transpose([1, 2, 0]).astype(np.uint8)
    #     imageio.imwrite('train1.jpg', lr)
    #     if (batch > 2):
    #         break

    model = model.Model(args, checkpoint)
    # lr = torch.randn(4,3,6,6).to("cuda:0")
    # out = model(lr)
    # print(out.shape)

    t = Trainer(args, loader, model, loss, checkpoint)

    while not t.terminate():
        t.train()
        #t.test()

    checkpoint.done()