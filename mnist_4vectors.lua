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
local b1 = nn.Sequential()
local b2 = nn.Sequential()
local b3 = nn.Sequential()
local b4 = nn.Sequential()
b1:add(nn.Linear(layerSize[#layerSize], outputSize))
b2:add(nn.Linear(layerSize[#layerSize], outputSize))
b3:add(nn.Linear(layerSize[#layerSize], outputSize))
b4:add(nn.Linear(layerSize[#layerSize], outputSize))

local c1 = nn.Sequential()
local c2 = nn.Sequential()
local c3 = nn.Sequential()
local c4 = nn.Sequential()
local o1 = nn.Sequential()
local o2 = nn.Sequential()
local o3 = nn.Sequential()
local o4 = nn.Sequential()
local o5 = nn.Sequential()
local o6 = nn.Sequential()
c1:add(b1):add(nn.LogSoftMax()) 
c2:add(b2):add(nn.LogSoftMax()) 
c3:add(b3):add(nn.LogSoftMax()) 
c4:add(b4):add(nn.LogSoftMax()) 

o1:add(nn.ConcatTable()
            :add(b1)  
            :add(b2) ) :add(nn.PairwiseDistance()) 


o2:add(nn.ConcatTable()
            :add(b2)
            :add(b3) ) :add(nn.PairwiseDistance())


o3:add(nn.ConcatTable()
            :add(b1)
            :add(b3) ) :add(nn.PairwiseDistance())

o4:add(nn.ConcatTable()
            :add(b1)
            :add(b4) ) :add(nn.PairwiseDistance())


o5:add(nn.ConcatTable()
            :add(b2)
            :add(b4) ) :add(nn.PairwiseDistance())


o6:add(nn.ConcatTable()
            :add(b3)
            :add(b4) ) :add(nn.PairwiseDistance())

model:add(nn.ConcatTable()
            :add(c1)
            :add(c2)
            :add(c3)
            :add(c4)
            :add(o1)
            :add(o2)
            :add(o3)
            :add(o4)
            :add(o5)
            :add(o6)
)

model:cuda() --ship to gpu
print(tostring(model))
torch.save('modelTest.dat',model)
local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
-- print(model.modules)

-- criterion = nn.MultiMarginCriterion():cuda()
crit1  = nn.MultiMarginCriterion():cuda()
crit2  = nn.MultiMarginCriterion():cuda()
crit3  = nn.MultiMarginCriterion():cuda()
crit4  = nn.MultiMarginCriterion():cuda()
crito1 = nn.HingeEmbeddingCriterion():cuda()
crito2 = nn.HingeEmbeddingCriterion():cuda()
crito3 = nn.HingeEmbeddingCriterion():cuda()
crito4 = nn.HingeEmbeddingCriterion():cuda()
crito5 = nn.HingeEmbeddingCriterion():cuda()
crito6 = nn.HingeEmbeddingCriterion():cuda()
criterion = nn.ParallelCriterion():add(crit1):add(crit2):add(crit3):add(crit4):add(crito1):add(crito2):add(crito3):add(crito4):add(crito5):add(crito6)



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
    local A = torch.CudaTensor(16,1):fill(-1)
    -- print(model.modules)
    -- print('DEBUG - network  output:'); print(y) 
    -- print('DEBUG - expected output:') ; print(yt) 
    -- print('DEBUG - y[3]:') ; print(y[3]) 
    local err  = criterion:forward(y, {yt,yt,yt,yt,A,A,A,A,A,A})
    -- local err2 = crit3:forward(y[3], A)
    -- print('ERR2'); print(err2) 
    -- err = err + err2
    lossAcc = lossAcc + err
    confusion:batchAdd(y[1],yt)
    
    
    if train then
      function feval()
        model:zeroGradParameters() --zero grads
        local dE_dy  = criterion:backward(y,{yt,yt,yt,yt,A,A,A,A,A,A}) 
        -- print('BACK - output:'); print(dE_dy) 
        -- local dE_dy3 = crit3:backward(y[3], A)
        -- print('BACK - y[3]:') ; print(dE_dy3)         
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

---    ### Introduce momentum, L2 regularization
--reset net weights
model:apply(function(l) l:reset() end)

optimState = {
  learningRate = 0.5,
  momentum = 0.9,
  weightDecay = 1e-3
}
for e = 1, epochs do
  optimState.learningRate = optimState.learningRate * 0.98
  print('optimState.learningRate = ',optimState.learningRate)
  optimState.momentum = optimState.momentum * 0.99
  print('optimState.momentum=',optimState.momentum)
  optimState.weightDecay = optimState.weightDecay * 0.99
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

torch.save('model_mnist_4v.dat',model)
