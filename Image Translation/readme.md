Image-to-image translation takes an input image and outputs it in a different style.

This implementation is based on CycleGan. There is a generator model that uses a ResNet neural network that takes
an input image and generates the output. And there is discriminator model that uses a small CNN that predicts
whether an image is an original or a generated fake. Over the course of training the generator learns to
make better fakes by fooling the discriminator and the discriminator gets better at detecting fakes.

For this implementation I was trying to convert portraits to anime:

![alt text](https://raw.githubusercontent.com/tjbergstrom/ML/master/Image%20Translation/samples/1.jpg)
![alt text](https://raw.githubusercontent.com/tjbergstrom/ML/master/Image%20Translation/samples/1_generated.jpeg)

