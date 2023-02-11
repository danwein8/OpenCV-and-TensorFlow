import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

inception_v3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

classifier_model = inception_v3

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))
])

grace_hopper = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org'
                                                    '/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)

grace_hopper = np.array(grace_hopper)/255.0

result = classifier.predict(grace_hopper[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis=-1)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org'
                                                            '/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

predicted_class_name = imagenet_labels[predicted_class]
print("Prediction: {}".format(predicted_class_name))
# print(predicted_class_name.title())

video = cv.VideoCapture(0)

while True:
    success, image = video.read()
    image1 = cv.resize(image, IMAGE_SHAPE)
    image1 = np.array(image1)/255.0
    result = classifier.predict(image1[np.newaxis, ...])
    predicted_class = tf.math.argmax(result[0], axis=-1)
    predicted_class_name = imagenet_labels[predicted_class]
    cv.putText(image, predicted_class_name.title(), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    cv.imshow("Image", image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
