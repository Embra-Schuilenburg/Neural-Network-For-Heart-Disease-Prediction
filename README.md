# Project Documentation
Project: A heart disease classifier

Learning goals:
- Learn how to clean data
- Learn the basics of setting up, training, and testing a neural network

Data used: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
## Project reflection
I created this project by following a neural network creation tutorial found on freeCodeCamp() which can be found here: https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/. I then used publically available heart disease data linked above to train and test the neural network I created. This project was made easy by following tutorials but showed me how to use the nueral network theory I learned my undergraduate AI class. My favorite part was learning more about backpropagation, and some challenges I faced were:
1. confusion regarding how data should be split
2. errors regarding labeling in the backpropagation function
## Model iterations
First:
- 1 hidden layer with 6 units
- learning rate: .01
- 10000 epochs
- test accuracy: 88.33%
- <img src="https://github.com/Embra-Schuilenburg/Neural-Network-For-Heart-Disease-Prediction/blob/master/images/first%20Iteration%20confusion%20matrix" alt="First iteration's confusion matrix, 31 true negatives, 2 false negatives, 5 false positives, and 22 true positives" width="400"/>
Second (lower learning rate):
- 1 hidden layer with 6 units
- learning rate: .001
- 10000 epochs
- test accuracy: 60%
- <img src="https://github.com/Embra-Schuilenburg/Neural-Network-For-Heart-Disease-Prediction/blob/master/images/second%20iteration%20confusion%20matrix" alt="Second iteration's confusion matrix, 36 true negatives, 24 false negatives, 0 false positives, and 0 true positives" width="400"/>
Third (lower training epoch count):
- 1 hidden layer with 6 units
- learning rate: .01
- 5000 epochs
- test accuracy: 90%
- <img src="https://github.com/Embra-Schuilenburg/Neural-Network-For-Heart-Disease-Prediction/blob/master/images/third%20iteration%20confusion%20matrix" alt="Second iteration's confusion matrix, 36 true negatives, 24 false negatives, 0 false positives, and 0 true positives" width="400"/>
## Best model and interpretation of results
After experimenting for a while I found the model which I feel achieves the best results. 

Optimal parameters:
- 1 hidden layer with 2 units
- learning rate: .01
- 6000 epochs

### Results:
Test data classification accuracy: 91.67%

Test case table:

|       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 | 57 | 58 | 59 |
|--------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Predicted | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 1  | 0  | 0  | 0  | 0  | 0  | 0  | 1  | 1  | 0  | 1  | 1  | 1  | 0  | 1  | 0  | 1  | 1  | 0  | 0  | 0  | 0  | 1  | 0  | 0  | 0  | 1  | 0  | 1  | 1  | 0  | 1  | 1  | 0  | 0  | 1  | 0  | 1  | 0  | 0  | 0  | 1  | 0  | 0  | 1  | 1  | 0  | 0  | 1  | 1  |
| Actual    | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 1  | 0  | 1  | 1  | 1  | 0  | 1  | 0  | 1  | 1  | 0  | 0  | 0  | 0  | 1  | 0  | 0  | 0  | 1  | 0  | 1  | 1  | 0  | 1  | 1  | 0  | 0  | 1  | 0  | 1  | 0  | 0  | 0  | 1  | 1  | 0  | 1  | 1  | 0  | 0  | 1  | 0  |

Confusion matrix:

<img src="https://github.com/Embra-Schuilenburg/Neural-Network-For-Heart-Disease-Prediction/blob/master/images/first%20Iteration%20confusion%20matrix" alt="First iteration's confusion matrix, 33 true negatives, 2 false negatives, 3 false positives, and 22 true positives" width="400"/>	

### Unit activation quantities and variable:
- Input variable activations:

|            | age      | sex      | cp       | trestbps | chol     | fbs      | restecg  | thalach  | exang    | oldpeak  | slope    | ca       | thal     |
|------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Unit 1   | -0.09434 | -0.60031 | -0.54730 | -0.34553 | -0.15084 |  0.31233 | -0.11647 |  0.54805 | -0.52958 | -0.47194 | -0.23363 | -0.93946 | -0.82755 |
| Unit 2   |  0.08436 |  0.17338 |  0.17060 |  0.13593 |  0.07521 | -0.06359 |  0.08287 | -0.19320 |  0.19874 |  0.21568 |  0.12177 |  0.29131 |  0.25914 |

- Output unit activations:
  - Unit 1: -2.55
  - Unit 2: .8
 
### Interpretation
To conclude, here are some of my interpretations of the experiment. There were a few key points to point out for analysis:
1. The model responded positively to a shorter training time and smaller hidden layer size. 
2. The model was prone to overfitting (examples not included but occurred frequently when tuning the model).
3. Hidden unit activation from variables was polarized (variables which activated one unit deactivated the other and vice versa).

Based on this info here are my conclusions:
- The dataset is relatively simple (for the model to interpret) and the variables can be classified into categories of contributing to or against the likelihood of heart disease.
	- The biggest contributors the model classifying the patient as having heart disease were: high resting blood pressure on hospital admission, exercise inducing angina, ST depression induced by exercise relative to rest, a high number of major vessels colored by flourosopy, and a fixed defect or reversable defect result on a thalassemia test.
	- The biggest contributors the model classifying the patient as not having heart disease were: fasting blood sugar > 120 mg/dl and a high heart rate achieved during exercise testing.

## AI usage summary
I used chatGPT to ask me leading questions to guide my learning. I had consulted it for an intial outline of steps and guidance shaping the data. This project conatins modified code that was generated by chatGPT and I do not claim the code of this project as entirely my own. Furthermore I used chatGPT to suggest causes of error while debuging and used some of its recommended code to fix the errors. While I used chatGPT a bit more heavily than I would like to usually, the goal of this project was to learn and practice using knowledge from my courses in a real scenario. I'm happy with my behavior and results and feel I have completed those goals.
