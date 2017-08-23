-- This is a modified version of VGG network in
-- https://github.com/szagoruyko/cifar.torch
-- Modifications:
--  * removed dropout
--  * last nn.Linear layers substituted with convolutional layers
--    and avg-pooling
require 'nn'
require 'cudnn'
local function createModel(opt)
   local model = nn.Sequential()

   -- building block
   local function Block(nInputPlane, nOutputPlane)
      model:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1):noBias())
      model:add(cudnn.SpatialBatchNormalization(nOutputPlane,1e-3))
      model:add(cudnn.ReLU(true))
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
   if opt.dataset == 'cifar100' then
      num_classes = 100
   elseif opt.dataset == 'cifar10' then
      num_classes = 10
   end
   model:add(nn.Linear(512, num_classes))

    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
    end
    local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(0.5)
         v.bias:zero()
      end
    end
    ConvInit('cudnn.SpatialConvolution')
    BNInit('cudnn.SpatialBatchNormalization')

    for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
    model:cuda()
   return model
end

return createModel