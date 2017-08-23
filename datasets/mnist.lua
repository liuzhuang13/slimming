--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  CIFAR-10 dataset loader
--

local t = require 'datasets/transforms'

local M = {}
local MnistDataset = torch.class('resnet.MnistDataset', M)

function MnistDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
   self.opt = opt
end

function MnistDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function MnistDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire CIFAR-10 training set
local meanstd = {
   mean = {0},
   std  = {255.0},
}

function MnistDataset:preprocess()
   if self.split == 'train' and self.opt.da > 0 then
      return t.Compose{
         t.ColorNormalize(meanstd),
         -- t.HorizontalFlip(0.5),
         -- t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.ColorNormalize(meanstd)
   else
      error('invalid split: ' .. self.split)
   end
end

return M.MnistDataset
