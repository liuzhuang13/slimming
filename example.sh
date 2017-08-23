th main_bn.lua -netType vgg -dataset cifar10 -batchSize 64 -nEpochs 160 -save results/example_vgg_cifar10/ -S 0.0001
th prune/prune_bn_per.lua -model results/example_vgg_cifar10/model_160.t7 -per 0.7 -save results/example_vgg_cifar10/pruned/model_160_0.7.t7
th convert/build_vgg.lua -model results/example_vgg_cifar10/pruned/model_160_0.7.t7 -save results/example_vgg_cifar10/converted/model_160_0.7.t7
th main_fine_tune.lua -retrain results/example_vgg_cifar10/converted/model_160_0.7.t7 -dataset cifar10 -batchSize 64 -nEpochs 160 -save results/example_vgg_cifar10/fine_tune/
