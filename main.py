from gensim.models.keyedvectors import KeyedVectors
import unpickle
import pickle
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

    query = input("Welcome to Image Search! What would you like to search?\t")
    text_semantic = embed_text.se_text(query, glove, idfs)
    # pass it through the model
    display_images.display_imgs(image_ids, img_to_coco)

if __name__ == "__main__":
    main()
