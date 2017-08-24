require 'nn'
require 'cunn'
require 'cudnn'

opt = lapp[[
  --model        (default '')
  --save         (default '')
]]

model1 = torch.load(opt.model)
bn_index = {2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 51}
channel_index = {}
for i = 1, #bn_index do
  bn = model1:get(bn_index[i])
  channel_index[i] = torch.nonzero(bn.weight:resize(bn.weight:size()[1]):float())
  -- print(channel_index[i]:nElement())
  channel_index[i]:resize(channel_index[i]:size()[1])
end

model = nn.Sequential()

current = 1
 -- building block
 local function Block()
    -- print(bn_index[current])

    ind = channel_index[current]
    if current > 1 then
      last_ind = channel_index[current-1]
    end

    output_channel = ind:nElement()
    input_channel = current > 1 and channel_index[current-1]:nElement() or 3


    model:add(cudnn.SpatialConvolution(input_channel, output_channel, 3,3, 1,1, 1,1):noBias())
    model:add(cudnn.SpatialBatchNormalization(output_channel,1e-3))
    model:add(cudnn.ReLU(true))

    
    model:get(bn_index[current]).running_mean:copy(model1:get(bn_index[current]).running_mean:index(1, ind))
    model:get(bn_index[current]).running_var:copy(model1:get(bn_index[current]).running_var:index(1, ind))
    model:get(bn_index[current]).weight:copy(model1:get(bn_index[current]).weight:index(1, ind))
    model:get(bn_index[current]).bias:copy(model1:get(bn_index[current]).bias:index(1, ind))
    model:get(bn_index[current]).eps = model1:get(bn_index[current]).eps

    conv2_index = bn_index[current]-1

    if current > 1 then
      model:get(conv2_index).weight:copy(model1:get(conv2_index).weight:index(1, ind):index(2, last_ind))
    else
      model:get(conv2_index).weight:copy(model1:get(conv2_index).weight:index(1, ind))
    end
    current = current + 1
    return model
 end

 local function MP()
    model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
    return model
 end

 local function Group(ni, no, N, f)
    for i=1,N do
       Block(i == 1 and ni or no, no)
    end
    if f then f() end
 end

 Group(3,64,2,MP)
 Group(64,128,2,MP)
 Group(128,256,4,MP)
 Group(256,512,4,MP)
 Group(512,512,4)
 model:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
 model:add(nn.View(-1):setNumInputDims(3))

 model:add(nn.Linear(ind:nElement(), model1:get(model1:size()).bias:size()[1]))
 model:get(model:size()).weight:copy(model1:get(model1:size()).weight:index(2, ind))
 model:get(model:size()).bias:copy(model1:get(model1:size()).bias)

 model:cuda()

torch.save(opt.save, model)

print('Conversion is successful!')


