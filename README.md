# MusicalFace


**Musical Face is an app which lets you play songs from Spotify based on your facial expression.**





Your Expression             |  Neural Networks	       | Spotify songs based on expression |	
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/achyutapnK/MusicalFace/blob/master/images/happy_face.jpg)  |  ![](https://cdn-images-1.medium.com/max/1600/1*DW0Ccmj1hZ0OvSXi7Kz5MQ.jpeg)	|  ![](https://github.com/achyutapnK/MusicalFace/blob/master/images/Spotify_logo.png)









Musical Face brings a blend of images and audio to the end user of this application. Not everytime is it convinient for one to use voice/manual input to play songs based on mood. Musical Face uses sophisticated CNN and RNN models to play songs based on the expression of the user. 
A happy face unlocks 10 happy songs on Spotify for the user looking in his/her webcam.





![Workflow of Musical Face](https://github.com/achyutapnK/MusicalFace/blob/master/images/Workflow.png)






## Dataset
*The data source for images comes from a very common challenge from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) for Facial Expression Recognition of B&W images. 

*The song dataset comes from [MoodyLyrics: A Sentiment Annotated Lyrics Dataset](https://dl.acm.org/citation.cfm?id=3059340). This serves as data for training the RNN model.

## CNN Architecture for Facial Expression Recognition
The architectue consists of a CNN model, which is trained on 28,709 examples and validated on 3,589 examples, predicts the emotion of person as following:


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



![CNN Architecture](https://github.com/achyutapnK/MusicalFace/blob/master/images/model_8.png)







## RNN Architecture for Songs' mood classification based on lyrics

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


![RNN Architecture](https://github.com/achyutapnK/MusicalFace/blob/master/images/model_rnn_81.png)




The RNN model was trained on 1648 songs' lyrics and validated on 413 songs' lyrics. The model is then used for classification of 56000 songs' lyrics into the aforementioned 4 emotions.

The accuracy of CNN model reaches 61.49% and that of RNN is 84.5%.

**Files and usage:**

*train_cnn_face.py : training of CNN model for facial expression classification.
*train_RNN.py : training of RNN model for songs' emotion classification based on lyrics.
*prediction.py: file to be run from command prompt with usage as:
	
	python prediction.py --choice <your choice> --modelpath <path to saved cnn model>
	
choice: 

0: for webcam

1: for local saved image

modelpath: path to saved cnn model or exp_detector6149.h5 can be used for default.

**client_id** and **client_secret** should be provided from oneself and can be accessed by Spotify Developer Dashboard. Details can be found [here.](https://developer.spotify.com/documentation/general/guides/app-settings/#register-your-app)

Attached is a screenshot of the expected output after running prediction.py on the terminal:

![prediction.py](https://github.com/achyutapnK/MusicalFace/blob/master/images/prediction.png)

## References:
*[Convolutional Neural Networks for Facial Expression Recognition](http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf)


