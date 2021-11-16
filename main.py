import os,random,glob
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

import time

faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
faceRecognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

##########################################################################################################################

# Root folder of the dataset
faceDatasetFolder = 'celeb_mini'
# Label -> Name Mapping file
labelMap = np.load("celeb_mapping.npy", allow_pickle=True).item()

##########################################################################################################################

# Each subfolder has images of a particular celeb
subfolders = os.listdir(faceDatasetFolder)
# Let us choose a random folder and display all images
random_folder = random.choice(subfolders)
# Also find out the name of the celeb from the folder name and folder-> name mapping dictionary loaded earlier
celebname = labelMap[random_folder]
# Load all images in the subfolder
imagefiles = os.listdir(os.path.join(faceDatasetFolder, random_folder))
# Read each image and display along with the filename and celeb name
# for file in imagefiles:
# #     Get full path of each image file
#     fullPath = os.path.join(faceDatasetFolder,random_folder,file)
#     im = cv2.imread(fullPath)
#     plt.imshow(im[:,:,::-1])
#     plt.show()
# #     Also print the filename and celeb name
#     print("File path = {}".format(fullPath))
#     print("Celeb Name: {}".format(celebname))

##########################################################################################################################

index = {}
i = 0
faceDescriptors = None

for folder in subfolders:
    imagefiles = os.listdir(os.path.join(faceDatasetFolder, folder))

    for file in imagefiles:
        imagePath = os.path.join(faceDatasetFolder, folder, file)
        # print("processing: {}".format(imagePath))

        img = cv2.imread(imagePath)

        faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


        print("{} Face(s) found".format(len(faces)))
        # Now process each face we found
        for k, face in enumerate(faces):

            # Find facial landmarks for each detected face
            shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

             # convert landmarks from Dlib's format to list of (x, y) points
            landmarks = [(p.x, p.y) for p in shape.parts()]

            # Compute face descriptor using neural network defined in Dlib.
            # It is a 128D vector that describes the face in img identified by shape.
            faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

            # Convert face descriptor from Dlib's format to list, then a NumPy array
            faceDescriptorList = [x for x in faceDescriptor]
            faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
            faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

            # Stack face descriptors (1x128) for each face in images, as rows
            if faceDescriptors is None:
                faceDescriptors = faceDescriptorNdarray
            else:
                faceDescriptors = np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)

            # save the label for this face in index. We will use it later to identify
             # person name corresponding to face descriptors stored in NumPy Array
            index[i] = labelMap[folder]
            i += 1




#         print("{} Face(s) found".format(len(faces)))
#         for k, face in enumerate(faces):

#             shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

#             # landmarks = [(p.x, p.y) for p in shape.parts()]

#             faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

#             faceDescriptorList = [x for x in faceDescriptor]
#             # print(faceDescriptorList)
#             faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
#             faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

#             if faceDescriptor is None:
#                 faceDescriptors = faceDescriptorNdarray
#             else:
#                 faceDescriptors =  np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)

#             index[i] = labelMap[folder]
#             i += 1
        



        

# index = {}
# i = 0
# faceDescriptors = None

# for imagePath in imagePaths:
#   print("processing: {}".format(imagePath))
#   # read image and convert it to RGB
#   img = cv2.imread(imagePath)

#   # detect faces in image
#   faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#   print("{} Face(s) found".format(len(faces)))
#   # Now process each face we found
#   for k, face in enumerate(faces):

#     # Find facial landmarks for each detected face
#     shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

#     # convert landmarks from Dlib's format to list of (x, y) points
#     landmarks = [(p.x, p.y) for p in shape.parts()]

#     # Compute face descriptor using neural network defined in Dlib.
#     # It is a 128D vector that describes the face in img identified by shape.
#     faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

#     # Convert face descriptor from Dlib's format to list, then a NumPy array
#     faceDescriptorList = [x for x in faceDescriptor]
#     faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
#     faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

#     # Stack face descriptors (1x128) for each face in images, as rows
#     if faceDescriptors is None:
#       faceDescriptors = faceDescriptorNdarray
#     else:
#       faceDescriptors = np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)

#     # save the label for this face in index. We will use it later to identify
#     # person name corresponding to face descriptors stored in NumPy Array
#     index[i] = nameLabelMap[imagePath]
#     i += 1