
------   ### Define model and criterion

require 'nn'
require 'cunn'
require 'optim'
require 'gnuplot'
require 'trepl'
model = torch.load("model_2v.dat")
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(tostring(model))
print('--------- Fischer --------------')
-- 
print(model:get(25):get(1):get(1):get(1).weight[1][1])
print(model:get(25):get(3):get(1):get(1):get(1).weight[1][1])

-- print(model:get(6):get(3):get(1):get(1):get(1).weight:topk(3))
-- print(model:get(6):get(3):get(1):get(1):get(1).weight:topk(3,true))
-- print('-----------------------')
-- print(model:get(6):get(3):get(1):get(2):get(1).weight:topk(3))
-- print(model:get(6):get(3):get(1):get(2):get(1).weight:topk(3,true))
-- print('-----------------------')
--print(model:get(6):get(3):get(1):get(1):get(1):getParameters():topk(5))


--os.exit()


print('-----------------------')
--a,b = model:get(6):get(4):get(1):get(1):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(25):get(3):get(1):get(1):get(1).weight:sum(1):mul(1):sort()
print('vecA1:')
--print(a:reshape(256,1))-- , b:reshape(128,1),2)
--a,b = model:get(6):get(4):get(1):get(2):get(1).weight:sum(1):mul(10):abs():sort()
a,b = model:get(25):get(3):get(1):get(2):get(1).weight:sum(1):mul(1):sort()
print('vecA2:')
--print(a:reshape(256,1))

--a,b = model:get(24).weight:sum(2):mul(1):sort()
--print('vecR:')
--print(a:reshape(256,1))



print('-----------------------')
--print(model:get(6):get(3):get(1):get(2):get(1).weight:sum(1):mul(-10):sort())
print('-----------------------')

-- print(model:get(6):get(3):get(1):get(2):get(1):getParameters():topk(5))
print('-----------------------')


--local range = torch.range(1, 10)

-- gnuplot.pngfigure('last_layer.png')
-- gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
-- gnuplot.xlabel('epochs')
-- gnuplot.ylabel('Loss')
-- gnuplot.plotflush()
