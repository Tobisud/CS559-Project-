Python Files:<br>
<br>
<br>
capture_hands.py - functions for capturing a hand in a boundery box when it detect on Camera <br>
data_splitting.py - pre-processing raw dataset, run this before training a model. Split dataset into 3 train, valid & test. Change dir if need.<br>
model.py - get NetMobile-smallv3 model<br>
train.py - train model with the dataset in the folder. Check dir before traning & make sure it pointing to the correct dataset.<br>
test.py - using the test dataset to test the trained model<br>
predict.py- using the trained model to give prediction on 1 image. (for testing purpose only)<br>
webcame.py - main file, use camera to capture the hand gesture then use model to give a prediction. Currently set to return top 5 predictions for testing purpose. <br>
fine_tune.py - fine tune the model using our own dataset like different background or light setting. <br>
gesture_cnn.pth- trained model using the org data set<br>
hand_landmarker.task - use for mediapipe to capture human hand. <br>
<br>
<br>
Set up:<br>
Download the data set from:<br>
https://data.mendeley.com/datasets/j4y5w2c8w9/1-<br>
<br>
<br>
Step:<br>
You can train your own by:<br>
<br>
Run data_splitting.py to split the dataset and shuffle them. <br>
Run train.py to train the model.<br>
Run test.py to test the model with the test set.<br>
<br>
(Or download the gesture_cnn.pth instead)<br>
<br>
Run webcame.py to use live camera and test the prediction. <br>
