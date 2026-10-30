Python Files:
capture_hands.py - functions for capturing a hand in a boundery box when it detect on Camera
data_splitting.py - pre-processing raw dataset, run this 
before training a model. Split dataset into 3 train, valid & test. Change dir if need.
model.py - get NetMobile-smallv3 model
train.py - train model with the dataset in the folder. Check dir before traning & make sure it pointing to the correct dataset.
test.py - using the test dataset to test the trained model
predict.py- using the trained model to give prediction on 1 image. (for testing purpose only)
webcame.py - main file, use camera to capture the hand gesture then use model to give a prediction. Currently set to return top 5 predictions for testing purpose. 
fine_tune.py - fine tune the model using our own dataset like different background or light setting. 
gesture_cnn.pth- trained model using the org data set
hand_landmarker.task - use for mediapipe to capture human hand. 

Set up:
Download the data set from:
https://data.mendeley.com/datasets/j4y5w2c8w9/1-

Step:
You can train your own by:

Run data_splitting.py to split the dataset and shuffle them. 
Run train.py to train the model.
Run test.py to test the model with the test set.

(Or download the gesture_cnn.pth instead)

Run webcame.py to use live camera and test the prediction. 