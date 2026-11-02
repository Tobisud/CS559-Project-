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
<br>
<br>
Comparing CNN models:<br>
<br>
model.py supports: mobilenet_v3_small, resnet18, efficientnet_b0, efficientnet_b2, convnext_tiny<br>
<br>
python train.py --model efficientnet_b0<br>
python train.py --model resnet18<br>
python train.py --model mobilenet_v3_small<br>
<br>
Each run saves to results/&lt;model&gt;/ : model.pth, history.csv, training_loss.png, training_accuracy.png, run_info.txt<br>
Default is full fine-tuning (all layers train, backbone lr 1e-4, head lr 1e-3) with augmentation.<br>
<br>
python test.py --checkpoint results/efficientnet_b0/model.pth<br>
python test.py --checkpoint results/efficientnet_b0/model.pth --data webcam_test --name webcam<br>
<br>
test.py saves confusion matrix, classification report and metrics json (accuracy, top-5, macro F1, ms/image) next to the checkpoint.<br>
<br>
Ablations (use --tag so runs don't overwrite each other):<br>
python train.py --model efficientnet_b0 --unfreeze 0 --tag effb0_frozen  (only last layer trains, like the original)<br>
python train.py --model efficientnet_b0 --no-augment --tag effb0_noaug<br>
python train.py --model efficientnet_b0 --lr 3e-4 --backbone-lr 3e-5 --tag effb0_lowlr<br>
<br>
To use a new model in webcame.py / predict.py / fine_tune.py, set the model path to results/&lt;model&gt;/model.pth. The model type is read from the checkpoint, and the old gesture_cnn.pth still works.<br>
<br>
Note: if training is slow on CPU, try --unfreeze 3 or a smaller --epochs.<br>
<br>
Data split (updated):<br>
data_splitting.py now splits by person by default (train P1-P7, val P8, test P9-P10), so the test set is people the model never saw. Use the RAW images folder (ASL_Raw_Images/asl_dataset).<br>
python data_splitting.py --source "path\to\ASL_Raw_Images\asl_dataset" --dest "path\to\split"<br>
python train.py --model resnet18 --data "path\to\split" --workers 8<br>
Old random split (for comparison only): add --mode random --dest "path\to\split_random"<br>
