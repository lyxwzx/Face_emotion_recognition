The project of "Face emotion recognition" is my graduation project. 

The project can realize facial expression recognition in pictures and real-time.
Project implementation steps:
1. Tensorflow and Keras were used to train Fer2013 data, and the training model with high accuracy was saved in the form of H5.
2. Develop the PyCharm and deploy the trained model using the Flask. In this classifier, Python and Opencv (Haar Classifier) are used to accurately mark the face in real time, and return the predicted facial emotion and probability.
You can download the project, configure the project environment, and run.

This project includes “local upload facial expression recognition” module and “real-time dynamic facial expression recognition” module:
1. “local upload facial expression recognition” module：
Users can select pictures from local photo library files, or call the camera to save pictures after taking pictures, upload the selected pictures to the system, and the system will recognize the expressions and return the results on the interface.
2. “real-time dynamic facial expression recognition” module:
The images acquired per second are acquired through the video stream of the camera. Face detection and image preprocessing are carried out on the images per second using the Cascade classifier based on Haar features. Face expression recognition is carried out on the background of the system with the trained model.

You can download the project, configure the Flask and Python environment，run with Pycharm.



# “Face_emotion_recognition”项目是我的本科毕业设计。

项目可以在图片和实时动态视频里实现人脸表情识别。

项目实现步骤：
1. 使用Tensorflow、keras等对Fer2013数据进行训练，将准确率较高的训练模型保存为h5形式。
2. 在PyCharm进行开发，利用Flask部署训练好的模型。其中利用Python和Opencv (Haar classifier)能够实时准确地标记人脸，并返回预测的人脸情感和概率。

这个项目包括本地上传表情识别模块和实时动态表情识别模块:
1. 本地上传表情识别模块呈现效果：
使用者可以从本地图片库文件中选取图片，或者调用摄像头拍照后保存图片，将选取的图片上传到系统中，系统识别表情后，在界面上返回结果。
2. 实时动态表情识别模块呈现效果:
通过调取摄像头视频流获取每秒采集的图像，对每秒的图像利用基于Haar特征的Cascade分类器进行人脸检测并进行图像预处理操作，系统后台用已经训练好的模型对其进行人脸表情识别，最后，将识别结果及预测值返回在界面上。

您可以下载此项目，配置好项目环境，即可运行。
