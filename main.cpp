#include <dirent.h> // for opendir
#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/string.h>
#include <filesystem> // for ittering through the filesystem
#include <iostream>   // stdin and stdout
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <typeinfo>

using namespace cv;
using namespace std;
using namespace dlib;

// This code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example.
template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down =
    add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
using anet_type = loss_metric<fc_no_bias<
    128,
    avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<
        3, 3, 2, 2,
        relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

// function for inroling the images
void inrole_data(frontal_face_detector faceDetector,
                 shape_predictor shapePredictor, anet_type faceRecognizer);

// ----------------------------------------------------------------------------------------

// Main function
int main() {

  // testPoint
  cout << "Starting..." << endl;

  // Set up the faceDetector
  frontal_face_detector faceDetector = get_frontal_face_detector();
  // Set up the shapePredictor
  String face68 = "/home/matthew/DoppelGanger-Find-your-Celebrity-Look-Alike/"
                  "shape_predictor_68_face_landmarks.dat";
  shape_predictor shapePredictor;
  deserialize(face68) >> shapePredictor;
  // Set up the faceRecognizer
  anet_type faceRecognizer;
  deserialize("/home/matthew/DoppelGanger-Find-your-Celebrity-Look-Alike/"
              "dlib_face_recognition_resnet_model_v1.dat") >>
      faceRecognizer;

  // testPoint
  cout << "1 Successfully loading faceDetector, shapePredictor and "
          "faceRecognizer!"
       << endl;

  inrole_data(faceDetector, shapePredictor, faceRecognizer);

  // End ----------------------------------------------------------------

  // testPoint
  // cout << "number of face descriptors " << faceDescriptors.size() << endl;
  // cout << "number of face labels " << faceLabels.size() << endl;

  // testPoint
  cout << "Worked to the end of the file" << endl;

  return 0;
};

void inrole_data(frontal_face_detector faceDetector,
                 shape_predictor shapePredictor, anet_type faceRecognizer)
//     """This function creates a face descriptors of (1x128) for each face
//     in the images folder and names files and stores them as face descriptors
//     and labels.
{
  // create a dictionary to uses for containing file label and person name
  std::map<int, string> index;
  // create a vector for storing face descriptors
  std::vector<matrix<float, 0, 1>> faceDescriptors;

  std::cout << "directory_iterator" << endl;

  // Start --------------------------------------------------------------
  string celebFolder =
      "/home/matthew/DoppelGanger-Find-your-Celebrity-Look-Alike/celeb_mini";
  for (auto const &images : std::filesystem::directory_iterator{celebFolder})
    for (auto const &image : std::filesystem::directory_iterator{images}) {
      // load the image using the Dlib library
      matrix<rgb_pixel> img;
      load_image(img, image.path());

      // run face detection
      // Stoped Here ****************************************************
    };

  // End ----------------------------------------------------------------

  cout << "function worked" << endl;
}
