import os
import glob
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt


faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faceRecognizer = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)


def inrole_data(faceDetector, shapePredictor, faceRecognizer):
    index = {}
    i = 0
    faceDescriptors = None

    for images in os.listdir("celeb_mini"):
        imagefiles = os.listdir(os.path.join("celeb_mini", images))

        for image in imagefiles:
            imagePath = os.path.join("celeb_mini", images, image)

            img = cv2.imread(imagePath)
            imDli = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = faceDetector(imDli)

            # Now process each face we found
            for face in faces:
                # Find facial landmarks for each detected face
                shape = shapePredictor(imDli, face)
                # Compute face descriptor using neural network defined in Dlib.
                faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

                # Convert face descriptor from Dlib's format to list, then a NumPy array
                faceDescriptorList = [x for x in faceDescriptor]
                faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
                faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

                # Stack face descriptors (1x128) for each face in images, as rows
                if faceDescriptors is None:
                    faceDescriptors = faceDescriptorNdarray
                else:
                    faceDescriptors = np.concatenate(
                        (faceDescriptors, faceDescriptorNdarray), axis=0
                    )

                # person name corresponding to face descriptors stored in NumPy Array
                index[i] = np.load("celeb_mapping.npy", allow_pickle=True).item()[
                    images
                ]
                i += 1

    np.save("index.npy", index)
    np.save("faceDescriptors.npy", faceDescriptors)


def lookalike(faceDetector, shapePredictor, faceRecognizer):
    #
    faceDescriptors = np.load("faceDescriptors.npy")
    index = np.load("index.npy", allow_pickle="TRUE").item()

    # read image
    testImages = glob.glob("test-images/*.jpg")

    for image in testImages:
        im = cv2.imread(image)
        imDli = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        faces = faceDetector(imDli)

        for face in faces:
            shape = shapePredictor(imDli, face)

            faceDescriptor = faceRecognizer.compute_face_descriptor(im, shape)

            faceDescriptorList = [x for x in faceDescriptor]
            faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
            faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

            distances = np.linalg.norm(faceDescriptors - faceDescriptorNdarray, axis=1)

            argmin = np.argmin(distances)
            minDistance = distances[argmin]

            if minDistance <= 0.8:
                label = index[argmin]
            else:
                label = "unknown"

            celeb_name = label

            for images in os.listdir("celeb_mini"):
                imagefiles = os.listdir(os.path.join("celeb_mini", images))

                if (
                    np.load("celeb_mapping.npy", allow_pickle=True).item()[images]
                    == celeb_name
                ):
                    for image in imagefiles:
                        img_cele = cv2.imread(os.path.join("celeb_mini", images, image))
                        img_cele = cv2.cvtColor(img_cele, cv2.COLOR_BGR2RGB)
                        break

        plt.subplot(121)
        plt.imshow(imDli)
        plt.title("test img")

        plt.subplot(122)
        plt.imshow(img_cele)
        plt.title("Celeb Look-Alike={}".format(celeb_name))
        plt.show()


def main():

    if not os.path.exists("index.npy") or not os.path.exists("faceDescriptors.npy"):
        print("building face descriptors")
        inrole_data(faceDetector, shapePredictor, faceRecognizer)

    lookalike(faceDetector, shapePredictor, faceRecognizer)


if __name__ == "__main__":
    main()
