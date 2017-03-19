require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'dataAugmentation'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('../cifar.torch/cifar10-train.t7')
local testset = torch.load('../cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)
local bestSoFar = 1;

print(trainData:size())

saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
print(classes[trainLabels[100]]) -- display the 100-th image class

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using train values
for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--  ****************************************************************
--  Define our neural network
--  ****************************************************************
local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end


Block(3,74,5,5,1,1,2,2)
w, dE_dw = model:getParameters()
print('1. Number of parameters:', w:nElement())

Block(74,48,1,1)
w, dE_dw = model:getParameters()
print('2. Number of parameters:', w:nElement())

--Block(160,96,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(48,44,3,3,1,1,1,1)
w, dE_dw = model:getParameters()
print('3. Number of parameters:', w:nElement())

--Block(192,192,1,1)
Block(44,48,1,1)
w, dE_dw = model:getParameters()
print('4. Number of parameters:', w:nElement())

model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(48,40,3,3,1,1,1,1)
w, dE_dw = model:getParameters()
print('5. Number of parameters:', w:nElement())

Block(40,8,1,1)
w, dE_dw = model:getParameters()
print('6.Number of parameters:', w:nElement())

--Block(32,10,1,1)
--model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(8 * 8 * 8))
model:add(nn.Linear(512,256))
--model:add(nn.Linear(8 * 8 * 8, 10))
--w, dE_dw = model:getParameters()
local b1 = nn.Sequential()
local b2 = nn.Sequential()
b1:add(nn.Linear(256,10))
b2:add(nn.Linear(256,10))

local c1 = nn.Sequential()
local c2 = nn.Sequential()

local o1 = nn.Sequential()

c1:add(b1):add(nn.LogSoftMax())
c2:add(b2):add(nn.LogSoftMax())


o1:add(nn.ConcatTable()
            :add(b1)
            :add(b2) ) :add(nn.PairwiseDistance())


model:add(nn.ConcatTable()
            :add(c1)
            :add(c2)
            :add(o1)
)
print('7. Number of parameters:', w:nElement())

--model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
--model:add(nn.View(#classes):setNumInputDims(1))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4


--model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati

model:cuda()
--criterion = nn.ClassNLLCriterion():cuda()
crit1  = nn.MultiMarginCriterion():cuda()
crit2  = nn.MultiMarginCriterion():cuda()
crito1 = nn.HingeEmbeddingCriterion():cuda()
criterion = nn.ParallelCriterion():add(crit1,0.5):add(crit2,0.5):add(crito1,10)



w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

--local batchSize = 16
--img = trainData:narrow(1, 1, batchSize):cuda()
--model:forward(img)
--print(model.modules)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

epochs = 600
local batchSize = 16
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        --local err = criterion:forward(y, yt)
        local A = torch.CudaTensor(16,1):fill(-1)
        local err  = criterion:forward(y, {yt,yt,A})
    
--        if (i==1) then
  --        print('crit1: ', crit1:forward(y[1],yt))
    --      print('crit2: ', crit2:forward(y[2],yt))
      --    print('crit3: ', crito1:forward(y[3],A))
        --end
    
        lossAcc = lossAcc + err
        confusion:batchAdd(y[1],yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y, {yt,yt,A})
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError_2v.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

function MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nInputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

function FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
	 --v.weight:zero()
   end
end

---------------------------------------------------------------------

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

--initialize net weights
MSRinit(model)
FCinit(model)

fd = io.open('reproduce_2v.csv','w')
fd:write('epoch,train error, train loss, test error, test loss\n')
timer = torch.Timer()
for e = 1, epochs do
    print('Epoch ' .. e)
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    fd:write(e .. ',' .. trainError[e] .. ',' .. trainLoss[e] .. ',' .. testError[e] .. ',' .. testLoss[e] .. '\n')
    print('to csv: ' .. e .. ',' .. trainError[e] .. ',' .. trainLoss[e] .. ',' .. testError[e] .. ',' .. testLoss[e] .. '\n')
    if testError[e] < bestSoFar then
	torch.save('model_2v.dat',model)
	print('best network so far in epoc: ' .. e .. '\n')
	bestSoFar = testError[e]
    end
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end

    if (e==1) then
        print(model.modules)
    end
end
fd:close()
plotError(trainError, testError, 'Classification Error')


--  ****************************************************************
--  Network predictions
--  ****************************************************************


--model:evaluate()   turn off dropout

--print(classes[testLabels[10]])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10_2v.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end
