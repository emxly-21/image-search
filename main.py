from gensim.models.keyedvectors import KeyedVectors
import unpickle
import pickle
import sim
import numpy as np
import embed_text
import display_images
import make_database

def main():
    path = r"glove.6B.50d.txt.w2v"
    glove = KeyedVectors.load_word2vec_format(path, binary=False)

    resnet = unpickle.unpickle()
    # make_database.make_database()     # uncomment this only if you want to repickle the files

    # unpickle files
    with open("idfs1.pkl", mode="rb") as idf:
        idfs = pickle.load(idf)
    with open("img_to_caption1.pkl", mode="rb") as cap:
        img_to_caption = pickle.load(cap)
    with open("img_to_coco1.pkl", mode="rb") as coco:
        img_to_coco = pickle.load(coco)

    # uncomment this only if you want to repickle the image embeddings
    # img_embeddings = {}
    # weights = np.load("weight.npy")
    # bias = np.load("bias.npy")
    # for image in resnet:
    #     embedding = image*weights + bias
    #     img_embeddings[image] = embedding
    # with open('img_embeddings.pkl', mode='wb') as file:
    #     pickle.dump(img_embeddings, file)

    with open("img_embeddings.pkl", mode="rb") as file:
        img_embeddings = pickle.load(file)

    cos_sims = {}
    for x in img_embeddings:
        cos_sims[x] = sim.sim

    query = input("Welcome to Image Search! What would you like to search?\t")
    #display_images.display_imgs(image_ids, img_to_coco)

if __name__ == "__main__":
    main()
