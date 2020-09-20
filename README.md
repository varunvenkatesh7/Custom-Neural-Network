# Custom Neural Network

This is an informal framework that produces Neural Networks. They can any number of layers and neurons, so as to optimize efficiency.
The network uses backpropogation and stochastic gradient descent to learn.

The class is optimized to recognized handwritten digits from the MNIST dataset, but it can be implemented for any network, even NLP/Text classification as long as the inputs are vectorized.
It's a low-level abstraction that can serve similar purposes to ML libraries. It was built without using any ML libraries, just math and numpy(a little bit of cheating haha).
The data fed into data_loader is MNIST dataset packaged as a .targz file

I'm thinking about hosting it online in real time, so I'm working on getting user input data from drawings and normalizing it so that I can feed it into the network. I'm also implementing it in other classes. Unfortunately, my class schedule is tight, so progress is slow 

This was definitely the hardest project I've worked on, took me a while to understand deep learning and the calculus behind it.
