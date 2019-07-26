from gensim.models.keyedvectors import KeyedVectors
import embed_text
import sim
import mygrad as mg
from mygrad import Tensor
import json
import unpickle
import numpy as np
from mynn.optimizers.sgd import SGD
from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss
import cos_sim
import pickle

def main():
    path = r"glove.6B.50d.txt.w2v"
    glove = KeyedVectors.load_word2vec_format(path, binary=False)

    # loads the json file
    path_to_json = "captions_train2014.json"
    with open(path_to_json, "rb") as f:
        json_data = json.load(f)
    resnet = unpickle.unpickle()

    with open("idfs1.pkl", mode="rb") as idf:
        idfs=pickle.load(idf)
    with open("img_to_caption1.pkl", mode="rb") as cap:
        img_to_caption=pickle.load(cap)
    #with open("img_to_coco1.pkl", mode="rb") as coco:
        #img_to_coco=pickle.load(coco)
    model = Model()
    optim = SGD(model.parameters, learning_rate=0.1)

    for i in range(100):
        id1 = np.random.choice(list(resnet.keys()))
        #print(id1)
        id2 = np.random.choice(list(resnet.keys()))
        while id1 == id2:
            id2 = np.random.choice(list(resnet.keys()))

        #print(type(resnet[id1]),type(img_to_caption[id1][0]),type(resnet[id2]))
        good_image = mg.Tensor(resnet[id1])
        text = (img_to_caption[id1][0])
        bad_image = mg.Tensor(resnet[id2])

        sim_to_good = cos_sim.cos_sim(model(good_image), embed_text.se_text(text, glove, idfs))
        sim_to_bad = cos_sim.cos_sim(model(bad_image), embed_text.se_text(text, glove, idfs))



    # compute the loss associated with our predictions(use softmax_cross_entropy)
        loss = margin_ranking_loss(sim_to_good, sim_to_bad, 1, 0.1)
    # back-propagate through your computational graph through your loss
        loss.backward()
    # compute the accuracy between the prediction and the truth
        acc = accuracy(sim_to_good, sim_to_bad)
    # execute gradient descent by calling step() of optim
        optim.step()
    # null your gradients
        loss.null_gradients()
    print(acc)


class Model:
    def __init__(self):
        self.dense1 = dense(512, 50)

    def __call__(self, x):
        return self.dense1(x)

    @property
    def parameters(self):
        return self.dense1.parameters


def accuracy(simgood, simbad):
    return simgood-simbad


if __name__ == "__main__":
    main()