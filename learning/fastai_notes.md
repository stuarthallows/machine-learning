# Image classification

## Setup

Ensure all data files are in place in appropriate folders


## Data exploration (optional)

Run a pre-trained model with `precompute=True`, this is just so we can quicly examine a few images

Look at a few sample images


## Training

Choose a learning rate, look for highest learning rate where loss is clearly improving

Configure data augmentation

Train the last layers from precomputed activations (`precompute=True`) for 1-2 epochs  (augmentations will not work)

Set `precompute=False`

Fit the model, as precompute is off augmentations will function, train for 2-3 epochs with `cycle_len=1`

Save


### Fine tune the convolutional layers with learning rate annealing

If using model with number bigger than 34 on a dataset similar to ImageNet, i.e. side on photos of standard objects with size between 2000 and 500px)
`learn.bn_freeze(True) `

Unfreeze, so we can start updating the convolutional filters

Fit the model

Good options for cycle configurations;

| cycles | cycle_len | cycle_mult |
| -----: | --------: | ---------: |
|      3 |         1 |          2 |
|      3 |         2 |     _none_ |


> `cycle_len`: number of epochs between resetting the learning rate  
> `cycles`: number of times the learning rate is reset

If using model pre-trainined on imagenet data and training on similar data use
    `lrs=[1e-4, 1e-3, 1e-2]`

If using model pre-trainined on imagenet data and training on disimilar data use
    `lrs=[lr/9, lr/3, lr]`

> The more the test images differ from the images the model weights were generated on the greater the learning rate used for fine-tuning should be.

Consider plotting the loss over time
`learn.sched.plot_loss()`

Save


## Increase image size (optional)

There's no point increasing the size beyond the average image size.

Set data with larger image size

Freeze

`fit(lr, 3, cycle_len=1)`

Unfreeze

Find learning rate (optional)  
>When to find LR? Run once at start, maybe after unfreezing layers, also when changing the thing being trained or the way changing is done. Never any harm in running it.

`fit(lrs, 3, cycle_len=1)`

examine whether model underfitting or overfitting, if underfitting try

fit(1rs, 3, cycle_len=1, cycle_mult=2)

Save

If underfitting try
    `fit(1e-2, 1, cycle_mult=2)`


Then consider repeating for larger image sizes (64 => 128 => 256)


### Train on full model (optional)

Find the learning rate again

> If using differential learning rates the learning rate of the last layer is printed.

Train on the full network with `cycle_mult=2` until overfitting, then re-run with the number of epochs that works well


### Test time augmentation

Once the validation and training loss are about the same, i.e. no over or under-fitting

Run TTA


### Analyse

Plot confusion matrix

Examine images again having trained the model

* random correct
* random incorrect
* most correct from each class
* mose incorrect from each class


