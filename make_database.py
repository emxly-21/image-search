import json
import embed_text
import pickle

def make_database():
    # loads the json file
    path_to_json = "captions_train2014.json"
    with open(path_to_json, "rb") as f:
        json_data = json.load(f)
    print("json file loaded")

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
    print("created img_to_caption dictionary")

    # creates a Dict[img_ids: coco_url]
    for image in json_data['images']:
        img_to_coco[image['id']] = image['coco_url']

    for caption in range(414113):
        documents.append(json_data['annotations'][caption]['caption'])

    print("created img_to_coco dictionary")

    counters = [embed_text.to_counter(doc) for doc in documents]
    vocab = embed_text.to_vocab(counters)
    idfs = embed_text.to_idf(vocab, counters)

    print("created idfs dictionary")

    # Saves idfs into a pickle file
    with open('idfs1.pkl', mode='wb') as file:
        pickle.dump(idfs, file)
    # Saves img_to_caption into a pickle file
    with open('img_to_caption1.pkl', mode='wb') as file:
        pickle.dump(img_to_caption, file)
    # Saves img_to_coco into a pickle file
    with open('img_to_coco1.pkl', mode='wb') as file:
        pickle.dump(img_to_coco, file)
    print("pickle complete")