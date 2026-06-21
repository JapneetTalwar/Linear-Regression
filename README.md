# Linear-Regression
A simple program where you can plot lines and see a linear regression algorithm find the line of best fit!

<img width="477" height="377" alt="image" src="https://github.com/user-attachments/assets/f5ac395b-2b1d-4dc7-a704-ce464693ecfd" />

## Link to release
The github release is here: 
https://github.com/JapneetTalwar/Linear-Regression/releases/tag/v.1.0.0

To run, simply download the simulator.zip, extract the file, open it and then click on the simulator application.
- Download and extract Zip
- Run simulator application

## Main features
- clickable graph to input points that can be cleared at any time
- simple linear regression framework using pytorch
- Linear regression metrics
  - Epochs - Number of training runs the model has had
  - Training loss - The loss between estimated and real training values in raw pixelss calculated using Huber Loss
  - Test loss - The loss between estimated and real test values in raw pixels
  - R squared - A metric that ranks how close the model is to a line that perfectly crosses through all dots (from -1 to 1)
  - Equation of line in pixel values
 
## Running in python
If you are running the source code, you need to ensure that you have python 3, pygame and pytorch installed, and run the main simulator.py

## How it works
The Linear regressor is a simple torch network with 2 parameters, weight (w) and bias (b), and calculates a value by plugging the input (x) into the formula: $ wx + b$
The training data and test data is randomly chosen from the nodes you click on the graph and every 10 epochs the model displays it training progress.

## Credits and Acknowledgements
AI declaration: I used gemini for bugfixing and feedback, and it taught me a lot about normalisation
Credit to learnpytorch.io which I used to learn how to make a linear regressor in pytorch
Credit to pytorch and pygame libraries that I used for my project
