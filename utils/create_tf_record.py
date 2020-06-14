import tensorflow as tf
import os
import dataset_util
from PIL import Image
import numpy as np
import cv2
import random
from sklearn import model_selection
import pickle


def create_tfdatapoint(file_loc, file, labels):
    img = Image.open(os.path.join(file_loc, 'images', file))
    (width, height) = img.size
    encoded = tf.io.gfile.GFile(os.path.join(file_loc, 'images', file), "rb").read()
    encoded = bytes(encoded)
    image_format = b'png'
    filename = file.split('.')[0]
    data = np.genfromtxt(os.path.join(file_loc, 'labels', filename + '.txt'))
    data = data.reshape(int(data.size / 5), 5)

    classes = [int(x) for x in data[:, 0]]
    classes_text = [labels[x].encode('utf8') for x in classes]
    xmins = data[:, 1] - (data[:, 3] / 2.0)
    xmaxs = data[:, 1] + (data[:, 3] / 2.0)
    ymins = data[:, 2] - (data[:, 4] / 2.0)
    ymaxs = data[:, 2] + (data[:, 4] / 2.0)

    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(str.encode(filename)),
        'image/source_id': dataset_util.bytes_feature(str.encode(filename)),
        'image/encoded': dataset_util.bytes_feature(encoded),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_label_and_data


def create_record(file_loc, labels, outputPath, outputPath_val):
    # creates a classes.pbtx with the label info
    f = open('../data/classes.names', "w")
    # loop over the classes
    for k in labels:
        # construct the class information and write to file
        item = (str(labels[k]) + "\n")
        f.write(item)

    file_list = []
    print("[INFO] creating list of file name...")
    for file in os.listdir(os.path.join(file_loc, 'images')):
        file_list.append(file)

    random.shuffle(file_list)
    test_size = 0.20
    seed = 7
    train, test = model_selection.train_test_split(file_list, test_size=test_size, random_state=seed)
    pickle.dump(test, open('test_filelist.txt', 'wb'))
    pickle.dump(train, open('train_filelist.txt', 'wb'))

    writer = tf.io.TFRecordWriter(outputPath)
    writer_val = tf.io.TFRecordWriter(outputPath_val)

    for i, file in enumerate(train):
        print("[INFO] : train file no - {} converting...".format(i + 1))
        i += 1
        try:
            tf_example = create_tfdatapoint(file_loc, file, labels)
        except:
            print(file + "file not found")

        if tf_example == 0:
            print(file)
        else:
            writer.write(tf_example.SerializeToString())

    for i, file in enumerate(test):
        print("[INFO] : test file no - {} converting...".format(i + 1))
        i += 1
        try:
            tf_example = create_tfdatapoint(file_loc, file, labels)
        except:
            print(file + "file not found")
        if tf_example == 0:
            print(file)
        else:
            writer_val.write(tf_example.SerializeToString())

    writer.close()
    writer_val.close()

