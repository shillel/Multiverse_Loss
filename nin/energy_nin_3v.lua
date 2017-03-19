
------   ### Define model and criterion

require 'nn'
require 'cunn'

model = torch.load("model_3v.dat")
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(tostring(model))
print('--------- Representation singular values --------------')
print(model:get(24):get(4):get(1):get(1):get(1).weight:sum(1))




local repSize=192
local multiverseLayer=25
a,b = model:get(multiverseLayer):get(4):get(1):get(1):get(1).weight:sum(1):mul(1):abs():sort()
print('vecA1:')
print(a:reshape(repSize,1))
a,b = model:get(multiverseLayer):get(4):get(1):get(2):get(1).weight:sum(1):mul(1):abs():sort()
print('vecA2:')
print(a:reshape(repSize,1))

a,b = model:get(multiverseLayer):get(5):get(1):get(2):get(1).weight:sum(1):mul(1):abs():sort(2,true)
print('vecB2:')
print(a:reshape(repSize,1))


