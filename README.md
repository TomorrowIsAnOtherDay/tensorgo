# tensorgo
Using the tensorgo API for TensorFlow Async Model Parallel

<div align=center><img src="http://7xl3p7.com1.z0.glb.clouddn.com/17-9-7/82144936.jpg?imageMogr2/auto-orient/thumbnail/!70p"/></div>

The system is designed to be simple to use, while maintaining efficiency speedup and approximate model performence(may be better).
Three lines to transfer your model into a multi-gpu trainer.

```python
from tensorgo.train.multigpu import MultiGpuTrainer
from tensorgo.train.config import TrainConfig
# [Define your own model using initial tensorflow API]
bow_model = ...

train_config = TrainConfig(dataset=training_dataset, model=bow_model, n_towers=5, commbatch=1500)
trainer = MultiGpuTrainer(train_config)
probs, labels = trainer.run([model.prob, model.label], 
                            feed_dict={model.dropout_prob=0.2,
                                        model.bacth_norm_on=True})
```


# ToDo list
- [ ] add benchmark for image model, like cifar10 benchmark of official TF benchmak
- [ ] add unit test
- [ ] add model saver
- [x] add user-defined api for model output
- [x] sync all the parameters between workers and server before training
- [x] add feed\_dict api for dropout/batchnorm paramenters 

# Reference
- [Large Scale Distributed Deep Networks][1]
- [tensorpack][2]

[1]:http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf
[2]:https://github.com/ppwwyyxx/tensorpack/tree/master/tensorpack
