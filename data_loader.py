# data_loader loads in data from MNIST
import numpy as np
import _pickle as cPickle, gzip

def load_the_data():
    #returns tuple containing training data, validation data, test data
    # the MNIST database has 70k? images. training data = 50k entries, validation has 10k entries and test has 10k entries
    # training data contains a tuple of 2 ndarrays(one 784(28x28) array amnd one 10(0-9)(answer))
    # validation and test data contain a tuple of the ndarray and the digit
    file = gzip.open('C:/Users/Varun/Documents/PyProjects/DeepLearning/neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(file, encoding = 'latin1') #latin1 seems to be the only working format
    file.close()
    #reshaping data to the convenient format from the mnist data
    return (training_data, validation_data, test_data)

def load_the_data_wrap():
    tr_data, va_data, te_data = load_the_data()
    training_inputs=[]
    training_results=[]
    for x in tr_data[0]: #inputs
        training_inputs.append(np.reshape(x, (784, 1)))
    for y in tr_data[1]: #results
        training_results.append(result_as_vector(y))

    validation_inputs = []
    for x in va_data[0]: #inputs, resullts are kept as numbers
        validation_inputs.append(np.reshape(x, (784, 1)))
    
    test_inputs=[]
    for x in te_data[0]: #test inputs, results are kept as numbers
        test_inputs.append(np.reshape(x, (784, 1)))
    
    training_data = list(zip(training_inputs, training_results))
    validation_data = list(zip(validation_inputs, va_data[1]))
    test_data = list(zip(test_inputs, te_data[1]))


    return (training_data, validation_data, test_data)

def result_as_vector(x):
    newArray = np.zeros((10, 1))
    newArray[x] = 1.0
    return newArray