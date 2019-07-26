from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss
from mynn.optimizers.sgd import SGD
from embed_text import se_text
import numpy as np
import unpickle

class Model:
    def __init__(self):
        self.dense1 = dense(512, 50)

    def __call__(self, x):
        return self.dense1(x)

    @property
    def parameters(self):
        return self.dense1.parameters
import mygrad as mg
#from mygrad import Tensor
def accuracy(simgood,simbad):
    return simgood-simbad


id1=np.random.randint(0, 82784)
id2=np.random.randint(0, 82784)
while id1 == id2:
    id2=np.random.randint(0, 82784)
good_image = resnet[id1]
text = img_to_caption[id1]
bad_image =resnet[id2]
model=Model()
optim = SGD(model.parameters, learning_rate=0.1)

for epoch_cnt in range(10000):

    sim_to_good = sim(se_text(text), model(good_image))
    sim_to_bad = sim(se_text(text), model(bad_image))

    # compute the loss associated with our predictions(use softmax_cross_entropy)
    loss = margin_ranking_loss(sim_to_good, sim_to_bad, 1, 0.1)
    acc=accuracy(sim_to_good,sim_to_bad)
    # back-propagate through your computational graph through your loss
    loss.backward()

    # compute the accuracy between the prediction and the truth
    acc = accuracy(prediction, truth)

    # execute gradient descent by calling step() of optim
    optim.step()

    # null your gradients
    loss.null_gradients()