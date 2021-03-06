local mnist = require 'mnist';
local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


----- ### Shuffling data
function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------   ### Define model and criterion
require 'nn'
require 'cunn'
require 'optim'
batchSize = 16
epochs = 70
local inputSize = 28*28
local outputSize = 10
local layerSize = {inputSize,64,128}

model = nn.Sequential()
model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy
for i=1, #layerSize-1 do
	model:add(nn.Linear(layerSize[i], layerSize[i+1]))
	model:add(nn.ReLU())
end

model:add(nn.Linear(layerSize[#layerSize], outputSize))
model:add(nn.LogSoftMax())   -- f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)

model:cuda() --ship to gpu
print(tostring(model))
torch.save('modelTest.dat',model)
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement()) --over-specified model

---- ### Classification criterion
--criterion = nn.ClassNLLCriterion():cuda()
criterion = nn.MultiMarginCriterion():cuda()


--- ### Main evaluation + training function
function forwardNet(data, labels, train)
	timer = torch.Timer()

	--another helpful function of optim is ConfusionMatrix
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	local lossAcc = 0
	local numBatches = 0
	if train then
		--set network into training mode
		model:training()
	else
		model:evaluate()
	end
	for i = 1, data:size(1) - batchSize, batchSize do
		numBatches = numBatches + 1
		local x = data:narrow(1, i, batchSize):cuda()
		local yt = labels:narrow(1, i, batchSize):cuda()
		local y = model:forward(x)
		local err = criterion:forward(y, yt)
		lossAcc = lossAcc + err
		confusion:batchAdd(y,yt)
		
		--print('DEBUG - network  output:', y) 
		--print('DEBUG - expected output:', yt) 
		
		if train then
			function feval()
				model:zeroGradParameters() --zero grads
				local dE_dy = criterion:backward(y,yt)
				model:backward(x, dE_dy) -- backpropagation
			
				return err, dE_dw
			end
		
			optim.sgd(feval, w, optimState)
		end
	end
	
	confusion:updateValids()
	local avgLoss = lossAcc / numBatches
	local avgError = 1 - confusion.totalValid
	print(timer:time().real .. ' seconds')

	return avgLoss, avgError, tostring(confusion)
end

--- ### Train the network on training set, evaluate on separate set
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--### Introduce momentum, L2 regularization
--reset net weights
model:apply(function(l) l:reset() end)

optimState = {
	learningRate = 0.5,
	momentum = 0.9,
	weightDecay = 1e-3
}
for e = 1, epochs do
    optimState.learningRate = optimState.learningRate * 0.95
	print('optimState.learningRate = ',optimState.learningRate)
	optimState.momentum = optimState.momentum * 0.98
	print('optimState.momentum=',optimState.momentum)
	optimState.weightDecay = optimState.weightDecay * 0.98
	print('optimState.weightDecay=',optimState.weightDecay)
	trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
	trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
	testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
	
	if e % 5 == 0 then
		print('Epoch ' .. e .. ':')
		print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
		print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
		print(confusion)
	end
end

print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])


print('Test error mean: ',torch.mean(testError:narrow(1,testError:size(1)-10,10)));
print('Test error: ',testError)
print('Train error: ',trainError)
print('Test loss: ',testLoss)
print('Train loss: ',trainLoss)

torch.save('model_mnist_1v.dat',model)
