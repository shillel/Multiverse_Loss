local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

require 'nn'
require 'cunn'

model = torch.load("model_mnist_4v.dat")
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(tostring(model))

a,b = model:get(6):get(1):get(1):get(1).weight:sum(1):mul(10):sort()
print('vec1:')
print(a:reshape(128,1))-- , b:reshape(128,1),2)

a,b = model:get(6):get(2):get(1):get(1).weight:sum(1):mul(10):sort()
print('vec2:')
print(a:reshape(128,1))

a,b = model:get(6):get(3):get(1):get(1).weight:sum(1):mul(10):sort()
print('vec3:')
print(a:reshape(128,1))

a,b = model:get(6):get(4):get(1):get(1).weight:sum(1):mul(10):sort()
print('vec4:')
print(a:reshape(128,1))
