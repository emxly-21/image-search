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
from mynn.optimizers.adam import Adam

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

    model.dense1.weight = mg.Tensor(np.load('weight.npy'))
    model.dense1.bias = mg.Tensor(np.load('bias.npy'))
    optim = Adam(model.parameters)

    batch_size = 100
    for epoch_cnt in range(100):

        idxs = list(resnet.keys())
        np.random.shuffle(idxs)
        for batch_cnt in range(0, len(idxs) // batch_size - 1):
            batch_indices = idxs[(batch_cnt * batch_size):((batch_cnt + 1) * batch_size)]
            batch_indices2 = idxs[((batch_cnt + 1) * batch_size):((batch_cnt + 2) * batch_size)]
            # id1 = np.random.choice(list(resnet.keys()))
            # print(id1)
            id1 = batch_indices
            # while id1 == id2:
            id2 = batch_indices2

            # print(type(resnet[id1]),type(img_to_caption[id1][0]),type(resnet[id2]))
            good_image = resnet[id1[0]]
            bad_image = resnet[id2[0]]
            text = embed_text.se_text(img_to_caption[id1[0]][0], glove, idfs)
            for i in id1[1:]:
                good_image = np.vstack((good_image, resnet[i]))
                text = np.vstack((text, embed_text.se_text(img_to_caption[i][0], glove, idfs)))

            for i in id2[1:]:
                bad_image = np.vstack((bad_image, resnet[i]))

            sim_to_good = cos_sim.cos_sim(model(good_image), text)
            sim_to_bad = cos_sim.cos_sim(model(bad_image), text)

            # compute the loss associated with our predictions(use softmax_cross_entropy)
            loss = margin_ranking_loss(sim_to_good, sim_to_bad, 1, 0.1)
            # back-propagate through your computational graph through your loss
            loss.backward()

            # compute the accuracy between the prediction and the truth
            acc = accuracy(sim_to_good.data, sim_to_bad.data)
            # execute gradient descent by calling step() of optim
            optim.step()
            # null your gradients
            loss.null_gradients()


    np.save('weight', model.dense1.parameters[0].data)
    np.save('bias', model.dense1.parameters[1].data)


class Model:
    def __init__(self):
        self.dense1 = dense(512, 50)

    def __call__(self, x):
        return self.dense1(x)

    @property
    def parameters(self):
        return self.dense1.parameters


def accuracy(simgood, simbad):
    return np.mean(simgood - simbad > 0)


if __name__ == "__main__":
    main()