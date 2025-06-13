AI Project: Maze Navigation with A* Algorithm and Perceptron Learning
This repository contains the code for an Artificial Intelligence project developed as part of an academic course. The project consists of two main components: a maze navigation task using the A* search algorithm to guide an agent to multiple targets, and a machine learning task implementing Perceptron and Gradient Descent Delta Rule algorithms for binary classification on the Iris dataset. The code is implemented in Python using Jupyter Notebooks.
Project Overview
The project is divided into two distinct tasks:

Maze Navigation (Question 1):

An agent navigates a 20x20 maze to reach five target locations using the A* search algorithm.
The maze includes walls (obstacles), a starting position, and multiple target points.
The A* algorithm computes the shortest path to each target, prioritizing paths based on a cost function and Manhattan distance heuristic.
The agent moves sequentially to each target, updating its starting position to the last reached target, until all targets are visited.


Machine Learning (Tasks 1-6):

Implements the Perceptron Learning Rule and Gradient Descent Delta Rule for binary classification.
Uses the Iris dataset (Setosa vs. non-Setosa classes) to train and evaluate the models.
Includes data preprocessing, model training, accuracy evaluation, and visualization of errors over epochs.
Provides a detailed analysis of the algorithms, including challenges, activation functions, learning rate strategies, and dataset split impacts.



Features

Maze Navigation:

Custom MazeEnv environment with a 20x20 grid, walls, and targets.
A* algorithm with custom move costs (e.g., right: 4, left: 1, down: 3, up: 2) and Manhattan distance heuristic.
Path optimization to visit all five targets in a sorted order based on path cost.
Visualization of the maze using a graphical interface with color-coded elements (agent, walls, targets).
Tracks visited and explored states for performance analysis.


Machine Learning:

Implementation of Perceptron and Gradient Descent Delta Rule from scratch.
Data preprocessing: Loads Iris dataset, splits into 80/20 train-test sets, and handles binary classification.
Model training with error tracking over epochs.
Evaluation of model accuracy on the test set.
Visualization of misclassification errors over epochs using Matplotlib.
Detailed answers to questions about algorithm differences, activation functions, learning rates, dataset splits, and implementation challenges.



Technologies Used

Python: Core programming language for both tasks.
Jupyter Notebook: For interactive development and documentation.
Libraries:
NumPy: Numerical computations and array operations.
Matplotlib: Plotting errors over epochs.
Scikit-learn: Loading Iris dataset and splitting data.
Heapq: Priority queue for A* algorithm.
Collections (deque): Efficient queue operations.


Custom Modules:
agents: Provides base classes for agents and environments (assumed to be part of the course framework).



Repository Structure

Project_AI.ipynb: The main Jupyter Notebook containing all code, outputs, and answers to questions.
README.md: This file, providing an overview and instructions for the project.

How to Run

Prerequisites:

Install Python 3.12+.
Install required libraries:pip install jupyter numpy matplotlib scikit-learn


Ensure the agents module (likely provided by the course) is available in the project directory or Python path.


Setup:

Clone the repository:git clone <repository-url>
cd <repository-directory>


Open the Jupyter Notebook:jupyter notebook Project_AI.ipynb




Execution:

Run the notebook cells sequentially to execute the maze navigation and machine learning tasks.
The maze navigation section will:
Initialize the maze environment.
Add walls and visualize the maze.
Compute and follow paths to all targets using A*.


The machine learning section will:
Load and preprocess the Iris dataset.
Train Perceptron and Gradient Descent models.
Print weights, errors, and accuracies.
Plot errors over epochs.


Review the markdown cells for answers to questions about the machine learning tasks.



Sample Output

Maze Navigation:

Visualizes a 20x20 maze with the agent starting at (1,1) and targets at (10,1), (16,1), (18,1), (10,18), (16,18).
Outputs the sequence of moves (e.g., right, down, etc.), visited states, and explored states.


Machine Learning:
Task 1: Perceptron Learning Rule and Gradient Descent Delta Rule Implemented
Perceptron Weights: [-0.92  55.616 -22.118 157.46  68.234]
Gradient Descent Weights: [[0.02484322], [-1.49751655], [1.78245438], [0.62808771]]

Task 2: Iris dataset loaded
Features (first 5 rows):
 [[5.1 3.5 1.4 0.2]
  [4.9 3.  1.4 0.2]
  [4.7 3.2 1.3 0.2]
  [4.6 3.1 1.5 0.2]
  [5.  3.6 1.4 0.2]]
Targets (first 5 elements): [0 0 0 0 0]

Task 5: Model accuracies on test set:
Perceptron Accuracy: 0.3
Gradient Descent Accuracy: 0.3111111111111111


A plot comparing errors over epochs for Perceptron and Gradient Descent.



Limitations

Maze Navigation:

Assumes the agents module is available, which may not be included in the repository.
The A* algorithm uses fixed move costs, which may not reflect real-world scenarios.
No interactive control for the agent; paths are precomputed.


Machine Learning:

Low accuracy (0.3 for Perceptron, 0.311 for Gradient Descent) suggests potential issues with implementation or hyperparameter tuning.
Perceptron errors remain high and constant, indicating possible convergence issues.
Limited to binary classification on a subset of the Iris dataset.
No hyperparameter optimization (e.g., learning rate tuning).



Future Improvements

Maze Navigation:

Add interactive controls for manual agent movement.
Implement additional pathfinding algorithms (e.g., Dijkstra, BFS) for comparison.
Support dynamic obstacles or changing maze layouts.
Include path visualization in the graphical interface.


Machine Learning:

Debug and optimize Perceptron implementation to improve convergence.
Implement adaptive learning rate methods (e.g., AdaGrad, RMSprop).
Extend to multi-class classification using one-vs-rest or softmax.
Add cross-validation for more robust performance evaluation.
Include feature scaling or normalization to improve model performance.



Contributors

Developed as part of an Artificial Intelligence course project.

Acknowledgments

Thanks to the course instructors for providing the agents module and guidance.
The Iris dataset is sourced from scikit-learn, originally from the UCI Machine Learning Repository.

