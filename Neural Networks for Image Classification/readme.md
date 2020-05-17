### Neural Networks for Image Classification notes:

You can do binary classification - either it is or isn't an image of ___.

Or you can do categorical classification - classify what kinda scene an image is, or animals, etc.

No need to specify which. You need to have a dataset directory with one subdirectory for each of the classes, each filled with photos of course.

You can run a default train_a_model.py with zero arguments, or you can customize a model with several different preset tunings with the following optional arguments.

### Choose a model:

-m [model]

Options are cnn, lenet, vgg, deepnet

### Adjust pre-tuned learning rate optimizers:

-o [optimizer]

Options are "Adam" (time decay), "Adam2" (constant decay), "Adam3" (with amsgrad correction), 
"SGD" (low momentum), "SGD2" (medium momentum), "SGD3" (high momentum) (all SGD are with Nesterov),
"RMSprop", "Adadelta"

### Adjust the number of epochs:

-e [int]

### Adjust the batch size:

-b [size]

Options are xs, s, ms, m, lg, xlg

### Adjust the image/layer size:

-i [size]

Options are xs, s, m, lg, xlg

### Adjust the image processing augmentor:

-a [type]

Options are original, light1, ligh2, light3, medium1, medium2, medium3, heavy1, heavy2

### Example command:

python3 -W ignore -m lenet -e 75 -o Adam2 -b m -i m -a light
