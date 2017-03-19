local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

------   ### Define model and criterion

require 'nn'
require 'cunn'
require 'optim'
require 'gnuplot'

model = torch.load("model_mnist_3v.dat")
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(tostring(model))
print('--------- Fischer --------------')
-- print(model:get(6):get(3):get(1):get(1):get(1))
-- print(model:get(6):get(3):get(1):get(1):get(1).weight:topk(3))
-- print(model:get(6):get(3):get(1):get(1):get(1).weight:topk(3,true))
-- print('-----------------------')
-- print(model:get(6):get(3):get(1):get(2):get(1).weight:topk(3))
-- print(model:get(6):get(3):get(1):get(2):get(1).weight:topk(3,true))
-- print('-----------------------')
--print(model:get(6):get(3):get(1):get(1):get(1):getParameters():topk(5))
print('-----------------------')
--a,b = model:get(6):get(4):get(1):get(1):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(6):get(4):get(1):get(1):get(1).weight:sum(1):mul(10):sort()
print('vecA1:')
print(a:reshape(128,1))-- , b:reshape(128,1),2)
--a,b = model:get(6):get(4):get(1):get(2):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(6):get(4):get(1):get(2):get(1).weight:sum(1):mul(10):sort()
print('vecA2:')
print(a:reshape(128,1))

--a,b = model:get(6):get(5):get(1):get(1):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(6):get(5):get(1):get(1):get(1).weight:sum(1):mul(10):sort()
print('vecB1:')
print(a:reshape(128,1))
--a,b = model:get(6):get(5):get(1):get(2):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(6):get(5):get(1):get(2):get(1).weight:sum(1):mul(10):sort()
print('vecB2:')
print(a:reshape(128,1))

--a,b = model:get(6):get(6):get(1):get(1):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(6):get(6):get(1):get(1):get(1).weight:sum(1):mul(10):sort()
print('vecC1:')
print(a:reshape(128,1))
--a,b = model:get(6):get(6):get(1):get(2):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(6):get(6):get(1):get(2):get(1).weight:sum(1):mul(10):sort()
print('vecC2:')
print(a:reshape(128,1))



print('-----------------------')
--print(model:get(6):get(3):get(1):get(2):get(1).weight:sum(1):mul(-10):sort())
print('-----------------------')

-- print(model:get(6):get(3):get(1):get(2):get(1):getParameters():topk(5))
print('-----------------------')


local range = torch.range(1, 10)

-- gnuplot.pngfigure('last_layer.png')
-- gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
-- gnuplot.xlabel('epochs')
-- gnuplot.ylabel('Loss')
-- gnuplot.plotflush()
