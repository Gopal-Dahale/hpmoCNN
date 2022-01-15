#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <time.h>

#include <cstdio>
#include <string>

#include "neural_net.cuh"

void NeuralNet::save(std::string path)
{
  FILE *f = fopen(path.c_str(), "wb");
  if (!f)
  {
    printf("Failed to open file %s\n", path.c_str());
    return;
  }
  fwrite(this, sizeof(NeuralNet), 1, f);
  fclose(f);
}

void NeuralNet::load(std::string path)
{
  FILE *f = fopen(path.c_str(), "rb");
  if (!f)
  {
    printf("Failed to open file %s\n", path.c_str());
    return;
  }
  fread(this, sizeof(NeuralNet), 1, f);
  fclose(f);
}