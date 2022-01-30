#include "datasets/mnist224.cuh"

int MNIST224::reverse_int(int n) {
  const int bytes = 4;
  unsigned char ch[bytes];
  for (int i = 0; i < bytes; i++) {
    ch[i] = (n >> i * 8) & 255;
  }
  int p = 0;
  for (int i = 0; i < bytes; i++) {
    p += (int)ch[i] << (bytes - i - 1) * 8;
  }
  return p;
}

void MNIST224::read_images(std::vector<std::vector<unsigned char>> &train_images,
                           std::vector<std::vector<unsigned char>> &test_images) {
  int images_per_file = 2000;
  int num_train_files = min((int)(ceil(num_train / float(images_per_file))), 30);
  int num_test_files = min((int)(ceil(num_test / float(images_per_file))), 5);

  for (int i = 0; i < 2; i++) {
    int num_files = (i == 0 ? num_train_files : num_test_files);
    for (int j = 0; j < num_files; j++) {
      std::string filename;
      if (i == 0)
        filename = filename_train_images;
      else
        filename = filename_test_images;
      filename = filename + std::to_string(j) + ".idx3-ubyte";

      std::ifstream f(filename.c_str(), std::ios::binary);
      if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

      // read metadata
      int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
      f.read((char *)&magic_number, sizeof(magic_number));
      magic_number = reverse_int(magic_number);
      f.read((char *)&n_images, sizeof(n_images));
      n_images = reverse_int(n_images);
      f.read((char *)&n_rows, sizeof(n_rows));
      n_rows = reverse_int(n_rows);
      f.read((char *)&n_cols, sizeof(n_cols));
      n_cols = reverse_int(n_cols);

      for (int k = 0; k < n_images; k++) {
        std::vector<unsigned char> temp;
        temp.reserve(n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; j++) {
          unsigned char t = 0;
          f.read((char *)&t, sizeof(t));
          temp.push_back(t);
        }
        if (i == 0) {
          train_images.push_back(temp);
          if ((j * n_images + k + 1) >= num_train) break;
        } else {
          test_images.push_back(temp);
          if ((j * n_images + k + 1) >= num_test) break;
        }
      }
      f.close();
    }
  }
}

void MNIST224::read_labels(std::vector<unsigned char> &train_labels,
                           std::vector<unsigned char> &test_labels) {
  // read train/test labels
  for (int i = 0; i < 2; i++) {
    std::string filename;
    if (i == 0)
      filename = filename_train_labels;
    else
      filename = filename_test_labels;

    std::ifstream f(filename.c_str(), std::ios::binary);
    if (!f.is_open()) printf("Cannot read MNIST from %s\n", filename.c_str());

    // read metadata
    int magic_number = 0, n_labels = 0;
    f.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    f.read((char *)&n_labels, sizeof(n_labels));
    n_labels = reverse_int(n_labels);

    if (i == 0)
      n_labels = min(n_labels, num_train);
    else
      n_labels = min(n_labels, num_test);

    for (int k = 0; k < n_labels; k++) {
      unsigned char t = 0;
      f.read((char *)&t, sizeof(t));
      if (i == 0)
        train_labels.push_back(t);
      else
        test_labels.push_back(t);
    }
    f.close();
  }
}

void MNIST224::read_mnist_224() {
  std::vector<std::vector<unsigned char>> train_images, test_images;
  std::vector<unsigned char> train_labels, test_labels;

  read_images(train_images, test_images);
  read_labels(train_labels, test_labels);

  assert(train_images.size() == train_labels.size());
  assert(test_images.size() == test_labels.size());

  // Copy data from vector to array
  int input_size = rows * cols * channels;
  for (int k = 0; k < num_train; k++) {
    for (int j = 0; j < input_size; j++)
      f_train_images[k * input_size + j] = (float)train_images[k][j];
    f_train_labels[k] = (int)train_labels[k];
  }

  for (int k = 0; k < num_test; k++) {
    for (int j = 0; j < input_size; j++)
      f_test_images[k * input_size + j] = (float)test_images[k][j];
    f_test_labels[k] = (int)test_labels[k];
  }
}

void MNIST224::normalize() {
  float *mean_image;
  int input_size = rows * cols * channels;
  mean_image = (float *)malloc(input_size * sizeof(float));

  for (int i = 0; i < input_size; i++) {
    mean_image[i] = 0;
    for (int k = 0; k < num_train; k++) {
      mean_image[i] += f_train_images[k * input_size + i];
    }
    mean_image[i] /= num_train;
  }

  for (int i = 0; i < num_train; i++) {
    for (int j = 0; j < input_size; j++) {
      f_train_images[i * input_size + j] -= mean_image[j];
    }
  }

  for (int i = 0; i < input_size; i++) {
    mean_image[i] = 0;
    for (int k = 0; k < num_test; k++) {
      mean_image[i] += f_test_images[k * input_size + i];
    }
    mean_image[i] /= num_test;
  }

  for (int i = 0; i < num_test; i++) {
    for (int j = 0; j < input_size; j++) {
      f_test_images[i * input_size + j] -= mean_image[j];
    }
  }
}