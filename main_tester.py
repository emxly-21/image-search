from gensim.models.keyedvectors import KeyedVectors
import embed_text
import sim
import json
import unpickle
import numpy as np
from mynn.optimizers.sgd import SGD
from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss


def main():
    path = r"glove.6B.50d.txt.w2v"
    glove = KeyedVectors.load_word2vec_format(path, binary=False)

    # loads the json file
    path_to_json = "captions_train2014.json"
    with open(path_to_json, "rb") as f:
        json_data = json.load(f)
    resnet = unpickle.unpickle()

    documents = []
    img_to_caption = {}
    img_to_coco = {}

    # creates a Dict[img_ids: captions]
    for caption in json_data['annotations']:
        img_id = caption['image_id']
        if img_id in img_to_caption:
            img_to_caption[img_id].append(caption['caption'])
        else:
            img_to_caption[img_id] = []

     #creates a Dict[img_ids: coco_url]
    for image in json_data['images']:
        img_to_coco[image['id']] = image['coco_url']
    for caption in range(82783):
        documents.append(json_data['annotations'][caption]['caption'])

    counters = [embed_text.to_counter(doc) for doc in documents]
    vocab = embed_text.to_vocab(counters)
    idfs = embed_text.to_idf(vocab, counters)



    model = Model()
    optim = SGD(model.parameters, learning_rate=0.1)
    batch_size = 100

    for i in range(100):
        id1 = np.random.choice(list(resnet.keys()))
        print(id1)
        id2 = np.random.choice(list(resnet.keys()))
        while id1 == id2:
            id2 = np.random.choice(list(resnet.keys()))
        good_image = resnet[id1]
        text = img_to_caption[id1][0]
        bad_image = resnet[id2]
        sim_to_good = sim.sim(embed_text.se_text(text, glove, idfs), model(good_image))
        sim_to_bad = sim.sim(embed_text.se_text(text, glove, idfs), model(bad_image))

    # compute the loss associated with our predictions(use softmax_cross_entropy)
        loss = margin_ranking_loss(sim_to_good, sim_to_bad, 1, 0.1)
        acc = accuracy(sim_to_good, sim_to_bad)
    # back-propagate through your computational graph through your loss
        loss.backward()

    # compute the accuracy between the prediction and the truth

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
import mygrad as mg
#from mygrad import Tensor
def accuracy(simgood,simbad):
    return simgood-simbad

if __name__ == "__main__":
    main()