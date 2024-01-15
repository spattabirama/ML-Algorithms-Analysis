# Machine Learning Algorithms Analysis

Here I will be analysing the performance of kNN, Random Forest and Neural Network on 4 classic datasets.

## Datasets

I will be analyzing four datasets.

1. The Hand-Written Digits Recognition Dataset

The goal, here, is to analyze 8x8 pictures of hand-written digits (see, e.g., Fig. 1) and classify them as
belonging to one of 10 possible classes. Each class is associated with one particular digit: 0, 1, . . . , 9. In this
dataset, each instance is composed of 8 × 8 = 64 numerical attributes, each of which corresponding to the
grayscale value of one pixel in the image being classified. This dataset is composed of 1797 instances.

2. The Titanic Dataset

On April 15, 1912, the largest passenger liner ever made collided with an iceberg during her maiden voyage.
When the Titanic sank, it killed 1502 out of 2224 passengers and crew. One of the reasons the shipwreck
resulted in such loss of life was that there were not enough lifeboats for the passengers and crew. Although
luck was involved in surviving the sinking, some groups of people were more likely to survive than others.
The goal, here, is to predict whether a given person was likely to survive this tragedy. The dataset contains
data corresponding to 887 real passengers of the Titanic. Each row represents one person. The columns
describe different attributes of the person, including whether they survived, their age, their passenger-class,
sex, and the fare they paid. The target class (Survived) is encoded in the first column of the dataset. Notice that this dataset combines both numerical
and categorical attributes.

3. The Loan Eligibility Prediction Dataset

Here, the goal is to automatically (and accurately) predict whether a given person should qualify for a loan.
Each instance is described by 13 attributes, including the Loan ID, the gender of the applicant, whether the
applicant is married, their income, information about their credit history, etc. The binary target class to be
predicted is Loan Status: whether a given applicant’s request for a loan was approved or not. Notice that
this dataset contains 8 categorical attributes, 4 numerical attributes, and a Loan ID attribute. There is a
total of 480 instances in the dataset.

4.  The Oxford Parkinson’s Disease Detection Dataset

This dataset is composed of a range of biomedical voice measurements from 31 people, 23 of which are
patients with Parkinson’s disease. Each row in the dataset corresponds to the voice recording from one of
these individuals. Each attribute corresponds to the measurement of one specific property of the patient’s
voice—for example, their average vocal frequency or several measures of frequency variation. The goal, here,
is to predict whether a particular person is healthy or whether it is a patient with Parkinson’s. The binary
target class to be predicted is Diagnosis: whether the patient is healthy (class 0) or a patient with Parkinson’s
(class 1). There are 195 instances in this dataset. All 22 attributes are numerical. 

## Analysis

<img width="882" alt="final_table" src="https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/d7e3518f-d433-407d-88b9-6cf3b57f6207">

### The Hand-Written Digits Recognition Dataset

For hand-written digits dataset I used KNN and Neural network algorithm. Since the KNN is a simple
algorithm and neural Network is a complex and powerful one, I decided to analyse and compare the
performance of these two on this numerical dataset. I tried all 4 algorithms(others are RandomForest and
Multinomial Naive bayes) and these 2 were giving better performance.
Both Neural Network and k-NN seems to be performing better for numerical datasets over random forest.
Eventhough it is a big dataset, the relation between the instances and the target is fairly straight forward,
making neural network an easy choice as it also performs better than any algorithm. k-NN algorithm was
chosen over random forest because in this scenario, the population of instances with same digit will be quite
close together in the map and hence it is easier for k-NN to perform better than Random forest.

#### kNN Algorithm

The predictions of KNN on Digit dataset was time consuming but it gave an accuracy of 98.87% and
F1 score of 98.83 at k=7. In the case of Testing data,The accuracy ranges between 98-94 percentage whereas in training it was ranging 99-96. The model is already trained with training data and now
seeing the new unseen values and it fails to generalize the data which results in its low accuracy. Due
to this overfitting, the model accuracy is slightly lesser than training data. The accuracy gets better at a K value around 7-9 and drops again. I can see so many ups and downs
in the accuracy throughout the k values 1-51.

![Handwriting_test_KNN](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/0000df8c-82a2-40f1-b07d-7e0e98a44ba4)

#### Neural Network

As the
number of training samples fed into the network increases, it is generally expected that the cost, or
loss, decreases. The network gains more exposure to different variations in the dataset and is able to
generalize well.

1. Hyper Parameters: Since there were as much as 64 attributes, it was necessary to provide higher
value of regularization factor(lambda) to penalize attributes that did not contribute to the decision
of the class. So I chose a lambda value of 0.5. A value lower than that gave lower accuracy compared to 0.5. After varying step size from 1 to 0.001, an alpha of 0.1 gave the best results for
a stopping criterion of 500 iterations.
2. More layers: Adding more layers and increasing the complexity of the network did not give a
major change to both accuracy and f-score.
3. More neurons per layer: Increasing the neurons seemed to improve the performance of the model
to a great extent by being able to capture non-linear relationship within the dataset. Seems like
in this dataset, it is better to stick with one layer and higher number of neurons rather than
increasing the number of layers.
4. Chosen architecture: I would choose the architecture [64, 64, 10] as it covers all the points
mentioned above – one layer with higher number of neurons. This architecture provides the
maximum accuracy and f-score when compared to the others.

![DigitRecognition_NN_Plot](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/ad701778-855a-4293-b624-753495d6734b)

#### Random Forest

Stopping criteria used for all the graphs - minimal size for split criterion = 3 % of data (.03 * data size).

• Accuracy
The ntree value I am gonna choose here is 40, since it gives me the highest accuracy (almost 93.54 %)
among all other ntree values.

• F1 Score
The ntree value I am gonna choose here is 40, since it gives me the highest F1 score (almost 94.91 %)
among all other ntree values.

![DigitRecognition_RF_F](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/bc666c6f-868e-4406-9f25-108c31d5f22a)
![DigitRecognition_RF_A](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/2e45984a-0c21-4fbe-87bd-637fab7866d0)


### The Titanic Dataset

For Titanic dataset I decided to use k-NN, Neural network and Random forest since all the three are
performing equally given the right hyper parameters. Random Forest can handle high-dimensional data
effectively, even when there is a mixture of categorical and numerical features as it can select informative
features based on their importance, reducing the risk of overfitting and improving the model’s generalization.
So it performs better than both Neural network which could struggle with high dimentional data. Neural
Networks and k-NN also needs one-hot encoding to handle categorical and numerical features together. With
careful selection of hyper-parameters Neural network and k-NN still manages to get comparitively good
performance.

#### kNN Algorithm

The predictions of KNN on Titanic dataset was fast and gave an accuracy of 82.41% and F1 score of
81.1 at k=5. In the case of Testing data,The accuracy ranges between 77-82 percentage whereas in
training it was ranging 89-80. The model is already trained with training data and now seeing the
new unseen values and it fails to generalize the data which results in its low accuracy. Due to this
overfitting, the model accuracy is lesser than training data. The accuracy
gets better at a K value around 5 and drops again. I can see so many ups and downs in the accuracy
throughout the k values 1-51.

![Titanic_KNN_test](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/925f78f7-a3e8-48ff-989a-b37c9493d09d)

#### Neural Network

Eventhough as the number of training samples fed into the network increases, the cost decreases initially,
the network struggles to reduce it beyond a certain point and has fluctuations. It could be due to noisy
data and local minima.

1. Hyper Parameters: As the number of attributes were comparatively lesser than the previous
dataset, it was necessary to provide lower value of regularization factor(lambda) to provide more
importance to each of the attributes present. So I chose a lambda value of 0.1. A value lower
than that gave lower accuracy. After varying step size from 1 to 0.001, an alpha of 0.1 gave the
best results for a stopping criterion of 500 iterations.
2. More layers: Adding more layers and increasing the complexity of the network improved both
accuracy and f-score indicating that each of the attributes were interrelated and capturing that
improved the performance of the algorithm.
3. More neurons per layer: Increasing the neurons did not seem to provide fruitful results as
the number of attributes itself was less. It was important to establish the relation between the
attributes more, so increasing the number of neurons was not a good choice.
4. Chosen architecture: I would choose the architecture [9, 9, 4, 3, 2] as it covers all the points
mentioned above – multiple layers with lesser number of neurons as the layers progress. This
architecture provides the maximum accuracy and f-score when compared to the others.

![Titanic_NN_Plot](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/5b7ea7b8-47ee-4ab1-92c7-cf2d5123df75)

#### Random Forest

• Accuracy
The ntree value I am gonna choose here is 5, even though ntree=40 gives me the highest accuracy
(almost 82.86%) ,because of the training time I’ll prefer low ntree value which gives almost similiar
accuracy of 82.38%.

• F1 Score
The ntree value I am gonna choose here is 5, since it gives me the highest F1 score (almost 75.67 %)
among all other ntree values.

![Titanic-RF-F1](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/3aca267b-7a9b-4f69-a940-e5abf3639573)
![Titanic-RF-Accuracy](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/6d637de0-1de0-483f-a53a-c5008754d8f8)

### The Loan Eligibility Prediction Dataset

For Loan dataset also I decided to use Random forest, KNN and Neural network. Both Neural network
and Random forest are performing better when compared to k-NN for this dataset. Since the hyper parameters
and architecture was carefully chosen after rigorous trail and also since there are less number of attributes,
Neural network is able to provide a very good accuracy in comparison to the other algorithms. With more
layers it is able to understand the intricate relation between each of the attributes those generalizing well
on the test data. Random forest generally performs well on dataset with high dimensions and with both
categorical and numerical attributes when the right number of trees and stopping criterion is provided. The
reason for choosing the hyper parameters are explained in detail below.
Since Random forest algorithm was chosen for extra credit, the graphs and explanation are towards the
end of this document.
1) I ignored the Loan ID of the person in the loan dataset while training since it doesn’t contribute to
decide whether a person could get loan approved or not.

#### kNN Algorithm

The predictions of KNN on Loan dataset gave an accuracy of 80.2% and F1 score of 71 at k=19. In the
case of Testing data,The accuracy ranges between 73-80 percentage whereas in training it was ranging 100-80. The model is already trained with training data and now seeing the new unseen values and it
fails to generalize the data which results in its low accuracy. Due to this overfitting, the model accuracy
is lesser than training data. The accuracy gets better at a K value around
13 and 19 and drops again. I can see so many ups and downs in the accuracy throughout the k values
1-51.

![Loan_KNN_test](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/f3ed199c-d04a-49b0-9160-f9ef3744aaf2)

#### Neural Network

As
the number of training samples loaded into the network increases, the cost decreases initially which
is followed by a steep rise. This dip could be due to a local minima. After the second rise, the cost
reduces and stabilizes indicating it has reached the global minima.

1. Hyper Parameters: When higher values of the regularization parameter are applied to the Loan
dataset, the model’s performance deteriorates. The regularization penalty, which controls the
model’s complexity and prevents overfitting becomes too strong and suppresses the valuable
information from variety of features. Since each attribute in this dataset provides valuable
information, I gave a lambda value of 0.01 to give importance to them. After varying step size
from 1 to 0.001, an alpha of 0.01 gave the best results for a stopping criterion of 500 iterations.
Higher value of iterations only seemed to increase the computation time but not the performance
of the algorithm.
2. More layers: With two hidden layers the model gains the ability to learn and represent more
complex relationships between the input attributes and the target. This enables the network to
capture intricate patterns that may exist within the loan dataset.
3. More neurons per layer: Increasing the number of neurons per hidden layer has further improved
the performance of the neural network on the loan dataset. The network is capable of capturing
details and nuances in the dataset allowing it to extract more relevant information and make more
accurate predictions.
4. Chosen architecture: I would choose the architecture [21, 20, 12, 2] as it covers all the points
mentioned above – two layers with more number of neurons. This architecture provides the maximum accuracy and f-score when compared to the others.

![Loan_NN_Plot](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/6d90a9a4-3d68-47bd-86aa-4f504b21470d)

#### Random Forest

• Accuracy
To get the best accuracy, I would choose an ntree value of 40 as it gives the maximum accuracy of
80.79%.
Even though ntree = 50 also reaches accuracy same as ntree=40, ntree=40 takes less computation
and also gives the highest accuracy.

• F1 Score
To get the best F Score, I would choose an ntree value of 40 as it gives the maximum F Score of
87.54%.

![Loan_RF_F](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/797ad0bf-f77e-4649-93ba-eee15de6b8fd)
![Loan_RF_A](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/25253cc9-0d7a-45d1-8713-b63dfe38170f)

### The Oxford Parkinson’s Disease Detection Dataset

For parkinson’s dataset also I decided to use KNN and Neural network. Both Neural Network and
k-NN seems to be performing better for numerical datasets over random forest. Eventhough the number of
instances for this dataset is less, the complex relation between the instances and the target can be established
by neural network by increasing the number of layer and neurons. k-NN algorithm was chosen over random
forest because in this scenario, the population of instances with similar data will be quite close together in
the map and hence it is easier for k-NN to perform better than Random forest.
Since the KNN is a simple algorithm and neural Network is a complex and powerful one, I decided to
analyse and compare performance of these two on this dataset containing numcerical attributes.I tried all 4
algorithms(others are RandomForest and Multinomil Naive bayes) and these 2 were giving better performance.

#### kNN Algorithm

The predictions of KNN on Loan dataset gave an accuracy of 95.3% and F1 score of 93.91 at k=7.
In the case of Testing data,The accuracy ranges between 95-78 percentage whereas in training it was
ranging 100-80. The model is already trained with training data and now seeing the new unseen values
and it fails to generalize the data which results in its low accuracy. Due to this overfitting, the model
accuracy is lesser than training data. The accuracy gets better at a K value
around 13 and 19 and drops again. I can see so many ups and downs in the accuracy throughout the
k values 1-51.

![Titanic_KNN_test](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/781b3e8f-5ec0-4917-8c11-7918c2ef4f67)

#### Neural Network

The cost
gradually decreases as the number of training samples fed into the network increases. This is because a
larger training dataset provides more diverse examples for the network to learn from, allowing it to
apprehend a broader range of patterns.

1. Hyper Parameters: A value of 0.0001 for lambda applies a moderate amount of regularization to
the model. It strikes a balance by fitting the training data and preventing overfitting, allowing
the neural network to generalize well to unseen instances and effectively learn from the dataset’s
features.
2. More layers: Two-layer representation helps the network to find patterns and dependencies within
the Parkinsons dataset, contributing to improved accuracy and F1-score. One layer is too general
and is unable to establish deeper relationship between various attributes of the dataset.
3. More neurons per layer: Higher number of neurons per hidden layer has further enhanced the
performance of the neural network on the dataset. The increased neurons made the network more
flexible allowing it to understand the complex dataset to reach the final decision.
4. Chosen architecture: I would choose the architecture [22, 20, 16, 2] as it covers all the points
mentioned above – two layers with more number of neurons. This architecture provides the
maximum accuracy and f-score when compared to the others.

![Titanic_NN_Plot](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/2836d446-a7bb-44f7-9d91-2b02f67212c6)

#### Random Forest

• Accuracy
To get the best accuracy, I would choose an ntree value of 30 as it gives the maximum accuracy of
90%. Even though ntree = 40 also reaches accuracy same as ntree=30, ntree=30 takes less computation
and also gives the higher accuracy.
• F1 Score
To get the best F Score, I would choose an ntree value of 30 as it gives the maximum F Score of
93.8%.

![Titanic-RF-F1](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/57ceba56-d19c-4f9f-977d-3ece47bd5492)
![Titanic-RF-Accuracy](https://github.com/spattabirama/ML-Algorithms-Analysis/assets/124756255/19e1fe42-92f7-47ed-ac80-c6074f35b84e)


