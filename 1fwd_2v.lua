local mnist = require 'mnist';
local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
local mean = trainData:mean()
local std = trainData:std()

testData:add(-mean):div(std);

------   ### Define model and criterion

require 'nn'
require 'cunn'
require 'optim'
require 'gnuplot'

model = torch.load("model_mnist_2v.dat")
batchSize = 16

--1V: 
--criterion = nn.MultiMarginCriterion():cuda()

--2V:
crit1 = nn.MultiMarginCriterion():cuda()
crit2 = nn.MultiMarginCriterion():cuda()
crit3 = nn.HingeEmbeddingCriterion():cuda()
criterion = nn.ParallelCriterion():add(crit1):add(crit2):add(crit3)



--- ### Main evaluation + training function

function forwardNet(data, labels, train)
	--timer = torch.Timer()

	--another helpful function of optim is ConfusionMatrix
	--local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	--local lossAcc = 0
	local numBatches = 0
        model:evaluate()
        i  = 1	
	--for i = 1, data:size(1) - batchSize, batchSize do
		numBatches = numBatches + 1
		local x = data:narrow(1, i, batchSize):cuda()
		local yt = labels:narrow(1, i, batchSize):cuda()
		local y = model:forward(x)
                local A = torch.CudaTensor(16,1):fill(-1)

		--1V:
                --local err = criterion:forward(y, yt)
		--2V:
		local err  = criterion:forward(y, {yt,yt,A})


                --lossAcc = lossAcc + err
		--confusion:batchAdd(y,yt)
		
		--print('DEBUG - network  output:', y) 
		--print('DEBUG - err:', err)
                print(y[1][2])
                print(y[2][2])
                --a,b=y[5]:add(10):abs():sort()
                --print(a:reshape(10,1))
                --print(b:reshape(10,1))

                print('DEBUG - expected output:')
		print(yt) 
                
                --local p1=y[1]:add(10)
                --local p2=y[2]:add(10)
                --gnuplot.pngfigure('test.png')
                --gnuplot.plot({'p1',p1},{'p2',p2})
                --gnuplot.xlabel('classes')
                --gnuplot.ylabel('probability')
                --gnuplot.plotflush()
		
	--end
	
	--confusion:updateValids()
	--local avgLoss = lossAcc / numBatches
	--local avgError = 1 - confusion.totalValid
	--print(timer:time().real .. ' seconds')

	--return avgLoss, avgError, tostring(confusion)
end


--testLoss, testError, confusion = forwardNet(testData, testLabels, false)
forwardNet(testData, testLabels, false)	


--print('Test error: ',testError)
--print(confusion)
--[[
require 'gnuplot'

local range = torch.range(1, epochs)

gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()
]]--
