#!/usr/bin/env python3

from extract_face import extract_face
from face_embedding import get_embedding
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import datetime
import os
import cv2


detector = MTCNN()
facenet = load_model('facenet_keras.h5')
tolerance = 6

def get_embedding_from_image(image):
    face_pixels, box = extract_face(detector, image)
    if face_pixels is None:
        return None, None
    return get_embedding(facenet, face_pixels), box


def load_known_people(path):
    people_embeddings = {}
    i = 0

    for name in os.listdir(path):
        if name not in people_embeddings.keys():
            people_embeddings[name] = []

        if '.' in name:
            continue
        for file in os.listdir(path + name):
            print(file)
            suffix = file.split('.')[-1]
            if suffix not in ['png', 'jpg']:
                continue

            i += 1
            print(i)

            img_array = np.array(load_img(path + name + '/' + file))
            embeddings, _ = get_embedding_from_image(img_array)
            if embeddings is not None:
                people_embeddings[name].append(embeddings)
            else:
                print(name + "/" + file + " does not include any face.")

    return people_embeddings

known_people_embeddings = load_known_people('faces/')
print({name: len(embeddings) for name, embeddings in known_people_embeddings.items()})

def face_distance(face_encodings, face_to_compare):
    return np.linalg.norm(face_encodings - face_to_compare)

def embeddings_recognition(embeddings):
    min_distance, min_name = 0xFFFF, None
    for name, embeddings_list in known_people_embeddings.items():
        for i in range(len(embeddings_list)):
            distance = face_distance(embeddings_list[i], embeddings)
            print("distance to %s: %d" % (name, distance))
            if distance < min_distance:
                min_distance, min_name = distance, name
    if min_distance <= tolerance:
        return min_name, min_distance

    return None, 0

def draw_boxes(data, v_boxes, v_labels, v_distances):
    # load the image
    # data = pyplot.imread(filename)
    # plot the image
    pyplot.clf()
    pyplot.ion()
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        # y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        x1, y1, width, height = box
        # calculate width and height of the box
        # width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_distances[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.pause(0.025)
    pyplot.show()

from multiprocessing import Process, Queue
from queue import Empty

def collect_frames(q):
    print("collecting frames")

    cam = cv2.VideoCapture(0)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("fps %d" % fps)
    while True:
        ret, img = cam.read()
        if not ret:
            print("cannot read next frame")
            continue

        try:
            _ = q.get(False, 0)
        except Empty:
            q.put(img) # empty already, just put into it
        else:
            q.put(img) # not empty, but 'get' has removed item from queue, we can just put into it

def show_webcam(mirror=False):
    import multiprocessing as mp
    mp.set_start_method('spawn') # NOTE: this is important to make opencv compatible multi-process

    q = Queue(maxsize=1)
    webcam_proc = Process(target=collect_frames, args=(q, ))
    webcam_proc.start()

    # cam = cv2.VideoCapture(0)
    # fps = cam.get(cv2.CAP_PROP_FPS)
    while True:
        img = q.get()

        # ret_val, img = None, None
        # for _ in range(0, 10):
        #     cam.grab()
        # ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        print(img.shape)

        start = datetime.datetime.now()

        face_pixels, box = extract_face(detector, img)
        if face_pixels is None:
            print("no face found in the screen")
            boxes, names, distances = [], [], []
        else:
            embeddings = get_embedding(facenet, face_pixels)
            name, distance = embeddings_recognition(embeddings)
            boxes, names, distances = [box], [name], [distance]

        end = datetime.datetime.now()
        delta = end - start
        print("%d ms elapsed" % (int(delta.total_seconds() * 1000)))

        draw_boxes(img, boxes, names, distances)


if __name__ == '__main__':
    show_webcam()
