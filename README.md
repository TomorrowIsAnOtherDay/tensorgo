# tensorgo
Using the tensorgo API for TensorFlow Async Model Parallel

![](http://7xl3p7.com1.z0.glb.clouddn.com/17-9-7/11824570.jpg?imageMogr2/auto-orient/thumbnail/!50p)
<div align=center><img width="150" height="150" src="http://img.blog.csdn.net/20161028230559575"/></div>

The system is designed to be simple to use, while maintaining efficiency speedup and approximate model performence(may be better).
Three lines to transfer your model into a multi-gpu trainer.

```python
from tensorgo.train.multigpu import MultiGpuTrainer
from tensorgo.train.config import TrainConfig
# [Define your own model using initial tensorflow API]
bow_model = ...

train_config = TrainConfig(dataset=training_dataset, model=bow_model, n_towers=5, commbatch=50000/32)
trainer = MultiGpuTrainer(train_config)
trainer.run()
```
