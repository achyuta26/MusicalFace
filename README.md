# MusicalFace
Musical Face is an app which lets you play songs from Spotify based on your facial expression.

Musical Face brings a blend of images and audio to the end user of this application. Not everytime is it convinient for one to use voice/manual input to play songs based on mood. Musical Face uses sophisticated CNN and RNN models to play songs based on the expression of the user. 
A happy face unlocks 10 happy songs on Spotify for the user looking in his/her webcam.

The architectue consists of a CNN model, which is trained on 28,709 examples and validated on 3,589 examples, predicts the emotion of person as 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.


| Face Expression | Code |
| --- | --- |
| Angry | 0 |
| Disgust | 1 |
| Fear | 2 |
| Happy | 3 |
| Sad | 4 |
| Surprise | 5 |
| Neutral | 6 |

The architecture of the CNN is as follows:

Conv2D layers:
64, (3,3)
Dropout : 0.25
128, (5,5)
Dropout : 0.25
512, (3,3)
Dropout : 0.5
512, (3,3)
Dropout : 0.1
Activation for all Conv2D : tanh
Fully Connected Layers:
256, ReLU, Dropout(0.5)
512, ReLU, Dropout(0.5)
7, Softmax
BatchNormalization at all layers



Since the labelled dataset of songs had 4 emotions, angry, happy, sad and relaxed, the above 7 emotions were clubbed as follows:

|Face Expression|Clubbed Emotion for song|
|---|---|
| Angry, Disgust|Angry |
| Fear, Sad | Sad |
| Happy | Happy |
| Surprise, Neutral | Relaxed |


The architecture of RNN is as follows:

Embedding layer  - 	in (600) 
		out(600,50)
LSTM- 		in (600,50)
		out(32)
Dense-		in(32)
		Activation(ReLU)
		out(256)
		Dropout(0.2)
Dense-		in(256)
		Activation(Softmax)
		out(4)

The RNN model was trained on 1648 songs' lyrics and validated on 413 songs' lyrics. The model is then used for classification of 56000 songs' lyrics into the aforementioned 4 emotions.

The accuracy of CNN model reaches 61.49% and that of RNN is 84.5%.
