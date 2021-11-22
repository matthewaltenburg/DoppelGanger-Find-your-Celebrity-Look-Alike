#include <dirent.h> // for opendir
#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/string.h>
#include <iostream> // stdin and stdout
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

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
// function for managing directories
void listdir(string dirName, std::vector<string> &folderNames,
             std::vector<string> &fileNames, std::vector<string> &symlinkNames);

// function for managing files extensions
void filterFiles(string dirPath, std::vector<string> &fileNames,
                 std::vector<string> &filteredFilePaths, string ext,
                 std::vector<int> &imageLabels, int index);

// ----------------------------------------------------------------------------------------
// Main function
int main() {
  // Set up the faceDetector
  frontal_face_detector faceDetector = get_frontal_face_detector();
  // Set up the shapePredictor
  String face68 = "/home/matthew/DoppelGanger-Find-your-Celebrity-Look-Alike/"
                  "shape_predictor_68_face_landmarks.dat";
  shape_predictor landmarkDetector;
  deserialize(face68) >> landmarkDetector;
  // Set up the faceRecognizer
  anet_type net;
  deserialize("/home/matthew/DoppelGanger-Find-your-Celebrity-Look-Alike/"
              "dlib_face_recognition_resnet_model_v1.dat") >>
      net;

  cout
      << "Successfully loading faceDetector, shapePredictor and faceRecognizer!"
      << endl;

  // Start ----------------------------------------------------------------
  // Load data for enrollment
  string faceDatasetFolder =
      "/home/matthew/DoppelGanger-Find-your-Celebrity-Look-Alike/celeb_mini";
  std::vector<string> subfolders, fileNames, symlinkNames;
  listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

  return 0;
};

// Reads files, folders and symbolic links in a directory
void listdir(string dirName, std::vector<string> &folderNames,
             std::vector<string> &fileNames,
             std::vector<string> &symlinkNames) {
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(dirName.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir(dir)) != NULL) {
      // ignore . and ..
      if ((strcmp(ent->d_name, ".") == 0) || (strcmp(ent->d_name, "..") == 0)) {
        continue;
      }
      string temp_name = ent->d_name;
      switch (ent->d_type) {
      case DT_REG:
        fileNames.push_back(temp_name);
        break;
      case DT_DIR:
        folderNames.push_back(dirName + "/" + temp_name);
        break;
      case DT_LNK:
        symlinkNames.push_back(temp_name);
        break;
      default:
        break;
      }
      //   cout << temp_name << endl;
    }
    // sort all the files
    std::sort(folderNames.begin(), folderNames.end());
    std::sort(fileNames.begin(), fileNames.end());
    std::sort(symlinkNames.begin(), symlinkNames.end());
    closedir(dir);
  }
}

// filter files having extension ext i.e. jpg
void filterFiles(string dirPath, std::vector<string> &fileNames,
                 std::vector<string> &filteredFilePaths, string ext,
                 std::vector<int> &imageLabels, int index) {
  for (int i = 0; i < fileNames.size(); i++) {
    string fname = fileNames[i];
    if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos) {
      filteredFilePaths.push_back(dirPath + "/" + fname);
      imageLabels.push_back(index);
    }
  }
}