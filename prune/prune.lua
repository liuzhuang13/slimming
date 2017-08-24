require 'nn'
require 'cunn'
require 'cudnn'

opt = lapp[[
	--percent    (default 0.5)
	--model  (default '')
	--save   (default '')
]]
print(opt)

model = torch.load(opt.model)
model:cuda()
name = 'cudnn.SpatialBatchNormalization'

print(model)


total = 0
for k, v in pairs(model:findModules(name)) do
	total = total + v.weight:size(1)
end

bn = torch.zeros(total)
index = 1
for k, v in pairs(model:findModules(name)) do
	size = v.weight:size(1)
	bn:narrow(1, index, size):copy(v.weight:clone():float():abs())
	index = index + size
end

y, i = torch.sort(bn)
thre_index = math.floor(total * opt.percent)
thre = y[thre_index]


pruned = 0

for k,v in pairs(model:findModules(name)) do
	weight_copy = v.weight:clone()
	mask = weight_copy:abs():gt(thre):float():cuda()

	pruned = pruned + mask:size(1) - torch.sum(mask)

	v.weight:cmul(mask) 
	v.bias:cmul(mask)

	print('layer index: ', k)
	print('total channel: ', mask:size(1))
	print('remaining channel: ', torch.sum(mask), '\n')
end

pruned_ratio = pruned/total

torch.save(opt.save, model)

print('Successful!')