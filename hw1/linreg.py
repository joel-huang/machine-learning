import theano
import theano.tensor as tensor
import numpy as np

# load the csv data into a np array
csv = 'linear.csv'
data = np.genfromtxt(csv, delimiter=',')
print(data.shape) # let's confirm the dimensions of the ndarray

response = data[:,0] # first column is the response Y
features = data[:,1:] # last four columns are the feature matrix X

# define the model parameters
feature_dim = features.shape[1] # feature matrix dimensions
num_samples = features.shape[0] # number of training samples
learn_rate = .5

# init theano variables
x = tensor.matrix(name='x') # feature matrix
y = tensor.vector(name='y') # response vector

# init a updatable shared variable that holds the
# values of parameters we are going to optimize
w = theano.shared(np.zeros(shape=(feature_dim,1)), name='w')

# define the training loss and compute the loss gradient
loss = tensor.sum((tensor.dot(x,w).T - y)**2)/2/num_samples
grad = tensor.grad(loss, wrt=w)

# run the training
train_model = theano.function(inputs=[],
			outputs=loss,
			updates=[(w, w-learn_rate*grad)],
			givens={x:features, y:response})

for i in range(50):
    print(train_model())
print(w.get_value())
