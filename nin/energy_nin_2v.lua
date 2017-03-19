
------   ### Define model and criterion

require 'nn'
require 'cunn'

model = torch.load("model_2v.dat")
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(tostring(model))

-- 
print(model:get(25):get(1):get(1):get(1).weight[1][1])
print(model:get(25):get(3):get(1):get(1):get(1).weight[1][1])




print('-----------------------')
a,b = model:get(25):get(3):get(1):get(1):get(1).weight:sum(1):mul(1):sort()
print('vecA1:')
print(a:reshape(256,1))

a,b = model:get(25):get(3):get(1):get(2):get(1).weight:sum(1):mul(1):sort()
print('vecA2:')
print(a:reshape(256,1))
