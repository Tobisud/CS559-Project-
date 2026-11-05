Mediapipe approach:<br>
<br>
![alt text](image-1.png)<br>
<br>
MediaPipe detects 21 key points on a hand, including the wrist, finger joints, and fingertips.<br>
<br>
Each landmark has three coordinates:<br>
- x: Horizontal position.<br>
- y: Vertical position.<br>
- z: Relative depth of the landmark.<br>
The model predict a hand gesture base on the landmark instead of hand shape.<br>
<br>
Python Files:<br>
<br>
<br>
extract_landmarks.py - uses MediaPipe to detect and extract 21 hand landmarks from gesture images, normalizes their coordinates, and saves them to a CSV file for training hand gesture classification models. <br>
![alt text](image.png)<br>
<br>
capture_hands.py - functions for capturing a hand in a boundery box when it detect on Camera <br>
data_splitting.py - pre-processing raw dataset, run this before training a model. Split dataset into 3 train, valid & test. Change dir if need.<br>
model.py - get NetMobile-smallv3 model<br>
train_mediapipe.py - train model with the dataset in the folder. Check dir before traning & make sure it pointing to the correct dataset.<br>
model_mlp.py- simple MPL model<br>
webcame_Mediapipe.py - main file, use camera to capture the hand gesture then use model to give a prediction. Currently set to return top 5 predictions for testing purpose. <br>
gesture_mlp.pth- trained model using the org data set<br>
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
Run train_mediapipe.py to create the CSV file first.<br>
Run train_mediapipe.py to train the model.<br>
<br>
(Or download the gesture_mlp.pth instead)<br>
<br>
Run webcame_Mediapipe.py to use live camera and test the prediction. <br>

