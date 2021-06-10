import requests
# import http3
import os
import torch.distributed.rpc as rpc
import time

# client = http3.AsyncClient()

os.environ['MASTER_ADDR'] = '128.223.8.161'
os.environ['MASTER_PORT'] = '12366'

rpc.init_rpc("worker0", rank=0, world_size=5)

def prepare():
    return False
def load_data():
    return False
def train():
    return False

datasize = 58747
# list_servers = ['http://128.223.8.161:5000','http://128.223.8.162:5000','http://128.223.8.150:5000','http://128.223.8.151:5000']

preprations = []
data_loads = []

opt_vocab_embed1 = rpc.rpc_async("worker1", prepare, args=(),timeout=0)
opt_vocab_embed2 = rpc.rpc_async("worker2", prepare, args=(),timeout=0)
opt_vocab_embed3 = rpc.rpc_async("worker3", prepare, args=(),timeout=0)
opt_vocab_embed4 = rpc.rpc_async("worker4", prepare, args=(),timeout=0)

print('done')

train_dev1 = rpc.rpc_async("worker1", load_data, args=(0,datasize//4,opt_vocab_embed1.wait()[0],opt_vocab_embed1.wait()[1]),timeout=0)
train_dev2 = rpc.rpc_async("worker2", load_data, args=(datasize//4,2*datasize//4,opt_vocab_embed2.wait()[0],opt_vocab_embed2.wait()[1]),timeout=0)
train_dev3 = rpc.rpc_async("worker3", load_data, args=(2*datasize//4,3*datasize//4,opt_vocab_embed3.wait()[0],opt_vocab_embed3.wait()[1]),timeout=0)
train_dev4 = rpc.rpc_async("worker4", load_data, args=(3*datasize//4,4*datasize//4,opt_vocab_embed4.wait()[0],opt_vocab_embed4.wait()[1]),timeout=0)

begin = time.time()

ret_ld1 = rpc.rpc_async("worker1", train, args=(train_dev1.wait()[0], train_dev1.wait()[1], opt_vocab_embed1.wait()[0], opt_vocab_embed1.wait()[1], opt_vocab_embed1.wait()[2],0), timeout=0)
ret_ld2 = rpc.rpc_async("worker2", train, args=(train_dev2.wait()[0], train_dev2.wait()[1], opt_vocab_embed2.wait()[0], opt_vocab_embed2.wait()[1], opt_vocab_embed2.wait()[2],0), timeout=0)
ret_ld3 = rpc.rpc_async("worker3", train, args=(train_dev3.wait()[0], train_dev3.wait()[1], opt_vocab_embed3.wait()[0], opt_vocab_embed3.wait()[1], opt_vocab_embed3.wait()[2],0), timeout=0)
ret_ld4 = rpc.rpc_async("worker4", train, args=(train_dev4.wait()[0], train_dev4.wait()[1], opt_vocab_embed4.wait()[0], opt_vocab_embed4.wait()[1], opt_vocab_embed4.wait()[2],0), timeout=0)

print('round 0 of worker 1:', str(ret_ld1.wait()))
print('round 0 of worker 2:', str(ret_ld2.wait()))
print('round 0 of worker 3:', str(ret_ld3.wait()))
print('round 0 of worker 4:', str(ret_ld4.wait()))

end = time.time()

print('All training time: ' + str(end-begin))

rpc.shutdown()
