import torch
import apex

import timeit
import time
import numpy as np

import matplotlib.pyplot as plt
 

WIDTH = 1.0
ALPHA = 1.0

def run_func_cumulative(func, inp_t, *args):
  if len(args) == 0:
    out = func(inp_t)
  elif len(args) == 1:
    out = func(inp_t, args[0])

def bm_layer_cumulative(layer, inp_t, grad_t=None, backward=False):
  torch.cuda.nvtx.range_push("start")
  out_t = layer(inp_t) 
  if backward:
    if isinstance(layer, apex.parallel.SyncBatchNorm):
      out_t.grad_fn.apply(grad_t)
    else:
      out_t.grad_fn(grad_t)
    #out_t.backward(grad_t)
  torch.cuda.nvtx.range_pop()

def run_func(func, inp_t, *args):
  if len(args) == 0:
    out = func(inp_t)
  elif len(args) == 1:
    out = func(inp_t, args[0])
  torch.cuda.synchronize()

def bm_layer(layer, inp_t, grad_t=None, backward=False):
  torch.cuda.nvtx.range_push("start")
  out_t = layer(inp_t) 
  if backward:
    out_t.backward(grad_t)
  torch.cuda.synchronize()
  torch.cuda.nvtx.range_pop()

def build_layer(sync_bn=None, batch_size=2, space_size=32, feature_size=512, fp16=False):
  if sync_bn == 1:
    layer = apex.parallel.SyncBatchNorm(feature_size).cuda()
  elif sync_bn == 2:
    layer = torch.nn.BatchNorm1d(feature_size).cuda()
  elif sync_bn == 0:
    import syncbn
    layer = syncbn.welford_mean_var
  else:
    layer = None
  inp = torch.randn(batch_size, feature_size, space_size).requires_grad_().cuda()
  grad = torch.randn(batch_size, feature_size, space_size).cuda()
  if fp16:
    if sync_bn == 2 and not torch.backends.cudnn.enabled:
      layer = layer.half()
    inp = inp.half()
    grad = grad.half()
  return layer, inp, grad

class Timer:
  def __init__(self):
    self.data = np.array([])

  def new_item(self, t):
    self.data = np.append(self.data, t)

  def mean(self):
    return self.data.mean()
    
  def median(self):
    return np.percentile(self.data, 50)

def pyplot_draw(data, xticks, title, xlabel, ylabel, log, axis=None):
  plt.title(title)
  if axis != None:
    x = xticks
  else:
    x = range(len(xticks))
    plt.xticks(x, xticks)

  if log:
    plot = plt.plot
  else:
    plot = plt.loglog

  for key in data:
    plot(x, data[key], label=key, linewidth=WIDTH, alpha=ALPHA)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

def draw_plot(result, axis = None, title = None, xlabel = None, ylabel = None, log = False):
  data = {}
  xticks = []
  for item in next(iter(result.values())):
    data[item] = []
  for key in result:
    if axis != None:
      xticks.append(key[axis])
    else:
      xticks.append(key)
    for item in data:
      data[item].append( result[key][item] )
  if title:
    pyplot_draw(data, xticks, title, xlabel, ylabel, log, axis)
  print(xticks)
  print(data)

################################################################################
# batchnorm timing
################################################################################
fp16_flag = False
iteration = 11
#iteration = 1
configs = []
#configs.append((batch_size, feature_size, space_size))

#imagenet
#batch_size = 256
#configs.append((batch_size, 64, 12544))
#configs.append((batch_size, 64, 3136))
#configs.append((batch_size, 256, 3136))
#configs.append((batch_size, 128, 3136))
#configs.append((batch_size, 128, 784))
#configs.append((batch_size, 512, 784))
#configs.append((batch_size, 256, 784))
#configs.append((batch_size, 256, 196))
#configs.append((batch_size, 1024, 196))
#configs.append((batch_size, 512, 196))
#configs.append((batch_size, 512, 49))
#configs.append((batch_size, 2048, 49))

#adlr model (semantic segmentation)
batch_size = 2
configs.append((batch_size, 64, 160000))
configs.append((batch_size, 128, 40000))
configs.append((batch_size, 256, 40000))
configs.append((batch_size, 256, 10000))
configs.append((batch_size, 512, 10000))
configs.append((batch_size, 1024, 10000))
configs.append((batch_size, 2048, 10000))
configs.append((batch_size, 256, 1))

result = {}

torch.backends.cudnn.benchmark = True

for batch_size, feature_size, space_size in configs:
  print("start another one")

  result[(batch_size, feature_size, space_size)] = {}
  res = result[(batch_size, feature_size, space_size)]

  # warm up session
  with torch.backends.cudnn.flags(enabled=True):
    layer, inp, grad = build_layer(2, batch_size, space_size, feature_size, fp16_flag)
    bm_layer(layer, inp, grad, backward=True)
  
  torch.cuda.cudart().cudaProfilerStart()
  
  layer, inp, grad = build_layer(1, batch_size, space_size, feature_size, fp16_flag)
  bm_layer(layer, inp, grad, backward=True)
  print("warm up done")

  timer = Timer()
  for i in range(iteration):
    #timer.new_item(timeit.timeit('bm_layer(layer, inp, grad, backward=True)', number=1, globals=globals()))
    if i == 1:
      start = time.time()
    bm_layer_cumulative(layer, inp, grad, backward=True)
  #res['syncBN'] = timer.median()
  torch.cuda.synchronize()
  res['syncBN'] = time.time() - start

  print("timed syncBN")
  
  with torch.backends.cudnn.flags(enabled=True):
    layer, inp, grad = build_layer(2, batch_size, space_size, feature_size, fp16_flag)
    bm_layer(layer, inp, grad, backward=True)
    timer = Timer()
    for i in range(iteration):
      if i == 1:
        start = time.time()
      #timer.new_item(timeit.timeit('bm_layer(layer, inp, grad, backward=True)', number=1, globals=globals()))
      bm_layer_cumulative(layer, inp, grad, backward=True)
    #res['cudnn'] = timer.median()
    torch.cuda.synchronize()
    res['cudnn'] = time.time() - start

  print("timed cudnn")
  
  with torch.backends.cudnn.flags(enabled=False):
    layer, inp, grad = build_layer(2, batch_size, space_size, feature_size, fp16_flag)
    bm_layer(layer, inp, grad, backward=True)
    timer = Timer()
    start = time.time()
    for i in range(iteration):
      #timer.new_item(timeit.timeit('bm_layer(layer, inp, grad, backward=True)', number=1, globals=globals()))
      if i == 1:
        start = time.time()
      bm_layer_cumulative(layer, inp, grad, backward=True)
    #res['THCNN'] = timer.median()
    torch.cuda.synchronize()
    res['THCNN'] = time.time() - start

  print("timed THCNN")

  torch.cuda.cudart().cudaProfilerStop()

print("finished run!")

draw_plot(result, None, "syncBN", "size", "time", True)
plt.legend()
plt.show()
#draw_plot(result, axis = 1)
#========================end batchnorm timing===================================



'''
################################################################################
# reduce timing
################################################################################
iteration = 100

reduce_dim_base = 128
output_dim_base = 64

reduce_dim_iter = 10
output_dim_iter = 5



for i in range(output_dim_iter):
  output_dim = output_dim_base * (2**i)
  configs = []
  result = {}
  
  plt.subplot(output_dim_iter, 1, i+1)

  for j in range(reduce_dim_iter):
    reduce_dim = reduce_dim_base * (2**j)
    #configs.append((reduce_dim, output_dim))
    configs.append((reduce_dim, output_dim))
  
  for reduce_dim, output_dim in configs:
    result[(reduce_dim, output_dim)] = {}
    res = result[(reduce_dim, output_dim)]
  
    welford_fn, inp_t, grad_t = build_layer(0, 1, output_dim, reduce_dim)
  
    torch.cuda.cudart().cudaProfilerStart()
    timer = Timer()
    for i in range(iteration):
      timer.new_item(timeit.timeit('run_func(welford_fn, inp_t)', number=1, globals=globals()))
    res['welford'] = timer.median()
  
    inp_t = torch.randn(output_dim, reduce_dim).requires_grad_().cuda()
    timer = Timer()
    for i in range(iteration):
      timer.new_item(timeit.timeit('run_func(torch.mean, inp_t, 1)', number=1, globals=globals()))
    res['contig_dim_mean'] = timer.median()
    timer = Timer()
    for i in range(iteration):
      timer.new_item(timeit.timeit('run_func(torch.var, inp_t, 1)', number=1, globals=globals()))
    res['contig_dim_var'] = timer.median()
  
    inp_t = torch.randn(reduce_dim, output_dim).requires_grad_().cuda()
    timer = Timer()
    for i in range(iteration):
      timer.new_item(timeit.timeit('run_func(torch.mean, inp_t, 0)', number=1, globals=globals()))
    res['noncontig_dim_mean'] = timer.median()
    timer = Timer()
    for i in range(iteration):
      timer.new_item(timeit.timeit('run_func(torch.var, inp_t, 0)', number=1, globals=globals()))
    res['noncontig_dim_var'] = timer.median()
  
    torch.cuda.cudart().cudaProfilerStop()
  
  draw_plot(result, 0, "reduction_output_size_{0}".format(output_dim), "reduction_size", "median time")
  plt.legend()
plt.show()
#========================end batchnorm timing===================================
'''
