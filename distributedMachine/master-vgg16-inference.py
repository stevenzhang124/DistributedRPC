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

batch_size = 1
image_w = 224
image_h = 224

if __name__ == '__main__':
	# 初始化主节点的RPC连接
	rpc.init_rpc("master", rank=0, world_size=2)


	#generate random data
	inputs = torch.randn(batch_size, 3, image_w, image_h)
	out0 = stage0.Stage0(inputs)
	out1 = rpc.rpc_sync("worker1", stage1.Stage1, args=(out0, ))
	
	print(out1)

	# 关闭RPC连接
	rpc.shutdown()

	out = vgg16.VGG16Partitioned(inputs)
	print(out)




