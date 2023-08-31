#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <svm.h> // Make sure you have the libsvm headers available

int main() {
    // Set random seed for consistent results
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Generate random data
    const int num_samples = 20;
    const int num_features = 2;
    std::vector<svm_node> samples;
    std::vector<int> labels;

    for (int i = 0; i < num_samples; ++i) {
        svm_node sample[num_features + 1]; // +1 for the sentinel
        for (int j = 0; j < num_features; ++j) {
            sample[j].index = j + 1;
            sample[j].value = std::rand() / static_cast<double>(RAND_MAX);
        }
        sample[num_features].index = -1; // Sentinel
        samples.push_back(sample);

        // Assign a label based on a simple rule
        int label = (sample[0].value + sample[1].value) > 1.0 ? 1 : -1;
        labels.push_back(label);
    }

    // Train the SVM
    svm_problem problem;
    problem.l = num_samples;
    problem.x = &samples[0];
    problem.y = &labels[0];

    svm_parameter params;
    svm_parameter_init(&params);
    params.svm_type = C_SVC;
    params.kernel_type = RBF;
    params.C = 1.0;
    params.gamma = 0.5;

    svm_model* model = svm_train(&problem, &params);

    // Test the SVM
    int correct_predictions = 0;

    for (int i = 0; i < num_samples; ++i) {
        double prediction = svm_predict(model, samples[i]);
        int predicted_label = prediction > 0 ? 1 : -1;
        if (predicted_label == labels[i]) {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / num_samples * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // Clean up
    svm_free_and_destroy_model(&model);

    return 0;
}
