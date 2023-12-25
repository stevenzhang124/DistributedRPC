import os
import threading
import time
import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from vgg16.nodes_2 import stage0, stage1, vgg16

os.environ['GLOO_SOCKET_IFNAME'] = 'wlan0'
os.environ['TP_SOCKET_IFNAME'] = 'wlan0'
os.environ['MASTER_ADDR'] = '192.168.1.110' # 指定master ip地址
os.environ['MASTER_PORT'] = '7856' # 指定master 端口号


if __name__ == '__main__':
    # 初始化主节点的RPC连接
    rpc.init_rpc("worker1", rank=1, world_size=2)


	# 等待主节点的调用
    rpc.shutdown()

