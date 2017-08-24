# Prepare the directories to save the results
mkdir vgg_cifar10/
mkdir vgg_cifar10/pruned
mkdir vgg_cifar10/converted
mkdir vgg_cifar10/fine_tune

# The network slimming scheme is organized as a 4-stage pipeline in this implementation

# 1. Train vgg network with channel level sparsity, S is the lambda in the paper which controls the significance of sparsity
th main.lua -netType vgg -save vgg_cifar10/ -S 0.0001

# 2. Identify a certain percentage of relatively unimportant channels and set their scaling factors to 0
th prune/prune.lua -percent 0.7 -model vgg_cifar10/model_160.t7  -save vgg_cifar10/pruned/model_160_0.7.t7

# 3. Re-build a real compact network and copy the weights from the model in the last stage
th convert/vgg.lua -model vgg_cifar10/pruned/model_160_0.7.t7 -save vgg_cifar10/converted/model_160_0.7.t7

# 4. Fine-tune the compact network 
th main_fine_tune.lua -retrain vgg_cifar10/converted/model_160_0.7.t7 -save vgg_cifar10/fine_tune/
