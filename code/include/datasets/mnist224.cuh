#ifndef __MNIST224__
#define __MNIST224__
#include <cassert>
#include <fstream>
#include <string>
#include <vector>

class MNIST224 {
 private:
  int rows, cols, channels, num_train, num_test;
  std::string filename_train_images, filename_train_labels;
  std::string filename_test_images, filename_test_labels;
  int reverse_int(int);
  void read_images(std::vector<std::vector<unsigned char>> &,
                   std::vector<std::vector<unsigned char>> &);
  void read_labels(std::vector<unsigned char> &, std::vector<unsigned char> &);

 public:
  float *f_train_images, *f_test_images;
  int *f_train_labels, *f_test_labels;
  MNIST224(int num_train, int num_test, std::string path_train_images,
           std::string path_train_labels, std::string path_test_images,
           std::string path_test_labels)
      : rows(224),
        cols(224),
        channels(3),
        num_train(num_train),
        num_test(num_test),
        filename_train_images(path_train_images),
        filename_train_labels(path_train_labels),
        filename_test_images(path_test_images),
        filename_test_labels(path_test_labels) {
    int input_size = rows * cols * channels;
    f_train_images = (float *)malloc(num_train * input_size * sizeof(float));
    f_train_labels = (int *)malloc(num_train * sizeof(int));
    f_test_images = (float *)malloc(num_test * input_size * sizeof(float));
    f_test_labels = (int *)malloc(num_test * sizeof(int));
  };

  void read_mnist_224();
  void normalize();
};

#endif