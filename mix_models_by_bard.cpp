#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <math.h>

using namespace std;

int main() {
  // Get the list of model files.
  vector<string> model_files;
  for (auto &file : fs::directory_iterator("models")) {
    if (file.is_regular_file() && file.path().extension() == ".bin") {
      model_files.push_back(file.path().string());
    }
  }

  // Create a vector of models.
  vector<AIModel*> models;
  for (const auto &model_file : model_files) {
    // Load the model from the file.
    AIModel *model = new AIModel();
    model->load(model_file);

    // Add the model to the vector.
    models.push_back(model);
  }

  // Create the output model.
  AIModel *output_model = new AIModel();

  // Combine the models using a weighted average.
  vector<double> weights;
  for (int i = 0; i < models.size(); i++) {
    weights.push_back(1.0 / models.size());
  }
  for (int i = 0; i < models.size(); i++) {
    // Get the output of the i-th model.
    vector<double> output = models[i]->predict();

    // Add the output of the i-th model to the output of the output model, weighted by the weight.
    for (int j = 0; j < output.size(); j++) {
      output_model->add_output(output[j] * weights[i]);
    }
  }

  // Save the output model.
  output_model->save("myecoria.bin");

  // Delete the models.
  for (auto model : models) {
    delete model;
  }

  // Delete the output model.
  delete output_model;

  return 0;
}
