require 'nn'
require 'cunn'
require 'cudnn'
require 'saveTXT'

opt = lapp[[
	--path   (default 'results/1115_all0.5_0.00001')
	--iter 	 (default '160')
	--per    (default 0.5)
	--model  (default '')
	--save   (default '')
]]
print(opt)

-- model = torch.load(opt.path..'/model_'..opt.iter..'.t7'):cuda()
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

print(bn:size())
y, i = torch.sort(bn)
thre_index = math.floor(total * opt.per)
thre = y[thre_index]


pruned = 0

for k,v in pairs(model:findModules(name)) do
	weight_copy = v.weight:clone()
	mask = weight_copy:abs():gt(thre):float():cuda()

	pruned = pruned + mask:size(1) - torch.sum(mask)

	v.weight:cmul(mask) 
	v.bias:cmul(mask)

	-- print('total: ', mask:size(1))
	-- print('sum: ', torch.sum(mask))
	-- print(v.weight)
end

pruned_ratio = pruned/total

print('threshold: '..thre)
print('pruned: '..pruned_ratio)

torch.save(opt.save, model)