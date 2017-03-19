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

model = torch.load("model_mnist_1v.dat")
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(tostring(model))

print('-----------------------')
a,b = model:get(6).weight:sum(1):mul(-10):sort()
print('vecA:')
print(a:reshape(128,1))

