from gensim.models.keyedvectors import KeyedVectors
import embed_text
import json
import numpy as np

def main():
    path = r"glove.6B.50d.txt.w2v"
    glove = KeyedVectors.load_word2vec_format(path, binary=False)
    path_to_json = "captions_train2014.json"
    with open(path_to_json, "rb") as f:
        json_data = json.load(f)
    documents = []
    for caption in range(82783):
        documents.append(json_data['annotations'][caption]['caption'])
    counters = [embed_text.to_counter(doc) for doc in documents]
    vocab = embed_text.to_vocab(counters)
    idfs = embed_text.to_idf(vocab, counters)

if __name__ == "__main__":
    main()