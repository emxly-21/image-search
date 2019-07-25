def display_imgs(image_ids, ids_to_url):
    """
    Plots the images of the image_ids
    :param image_ids: an Iterable of ids
    :param ids_to_url: Dict[ids: coco_url)
    :return: None
    """
    for image in image_ids:
        url = ids_to_url[image]
        data = urlopen(url)

        # converting the downloaded bytes into a numpy-array
        url_format = url[url.rfind('.')+1:]
        img = plt.imread(data, format=url_format)  # shape-(460, 640, 3)

        # displaying the image
        fig, ax = plt.subplots()
        ax.imshow(img)