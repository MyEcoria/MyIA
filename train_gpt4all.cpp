#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

int main() {
  // Load the existing GPT4All model
  cout << "Loading model..." << endl;
  GPT4All model("gpt4all.bin");

  // Open the nano_node folder
  cout << "Opening nano_node folder..." << endl;
  fstream folder("nano_node");

  // Get all the files and folders in the folder
  vector<string> files;
  vector<string> folders;
  for (string file : folder) {
    if (file.find(".") != string::npos) {
      // This is a file
      files.push_back(file);
    } else {
      // This is a folder
      folders.push_back(file);
    }
  }

  // Create a progress bar
  int progress = 0;
  int total_files = files.size();
  auto start = std::chrono::high_resolution_clock::now();
  cout << "Learning files..." << endl;

  // Teach the model the names of the files and folders
  for (string file : files) {
    cout << "\rLearning file " << file << " " << progress << "/" << total_files << " " << std::flush;
    model.learn(file);
    progress++;
  }

  // Save the model
  cout << endl << "Saving model..." << endl;
  model.save("gpt4all.bin");

  // Print a message to the user
  cout << "Done! Elapsed time: " << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() << " seconds." << endl;

  return 0;
}
