import os
import sys
import base64
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
import time
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

from PIL import Image

# Some utilites
import numpy as np
from util import base64_to_pil

import cv2
import json

app = Flask(__name__)
graph = tf.get_default_graph()
# Model saved with Keras model.save()
MODEL_PATH = 'models/fer2013lyx61.h5'

faceCascade = cv2.CascadeClassifier('/Users/liyixin/PycharmProjects/Face/config/haarcascade_frontalface_alt.xml')
# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary


model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print('Model loaded. Start serving...')
label_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "suprise", 6: "neutral"}
result_real_time=""
probability_real_time=""

def predict_class(X_test):
    global graph
    with graph.as_default():
        predictions_class = model.predict_classes(X_test)

    return predictions_class


def predict_value(X_test):
    global graph
    with graph.as_default():
        predictions_value = model.predict(X_test)

    return predictions_value


def show_predict_result(prediction, predicted_probability):
    result = ""
    print('x_valid_image_norm predict result', label_dict[prediction[0]], "\n")
    for j in range(7):
        res = (label_dict[j] + ' probability:%1.9f' % (predicted_probability[0][j]))
        print(res)
        result += res+"\n"
    return result

# def model_predict(img, model):
#     # img = img.resize((224, 224))
#     img = img.resize((48, 48))
#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)
#
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='tf')
#     # # preds = model.predict(x)
#
#     return preds


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # print(request.json)  # data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4QAIRXhpZgAA/9sAQwAFAwQEBAM

        # # Get the image from post request
        # img = base64_to_pil(request.json)
        img = base64_to_pil(request.json).convert('L').resize((48, 48))
        img = np.asarray(img).reshape(-1, 48, 48, 1).astype('float32') / 255
        # print(img.shape)

        # # # Save the image to ./uploads
        # img.save("./uploads/image.png")
        begin = time.time()
        p_class = predict_class(img)
        predict_time1 = time.time()
        p_value = predict_value(img)
        predict_time2 = time.time()
        print("time:",str(predict_time1-begin),str(predict_time2-begin))
        print("预测结果(lable)：", p_class)
        print("预测结果：", label_dict[p_class[0]])
        print("每个预测值：", p_value)

        # 终端显示每个预测结果对应的预测值
        # probability_yang = show_predict_result(p_class, p_value)


        # prediction = model.predict_classes(img)
        # prediction_probability = model.predict(img)
        # print(prediction)
        # print(prediction_probability)

        # # # Make prediction
        # preds = model_predict(img, model)
        #
        # # Process your result for human
        # probability = "{:.3f}".format(np.amax(p_class[0]))  # Max probability
        # pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        #
        result = str(label_dict[p_class[0]])  # Convert to string
        result = result.replace('_', ' ').capitalize()
        # probability =str(p_value)
        # probability= json.dumps(p_value.item())
        # probability=p_value.tolist()

        # probability = str(p_value)  # Convert to string
        # probability = probability.replace('_', ' ').capitalize()
        probability = str(",".join(str(float(_*100)) for _ in p_value[0]))



        print(result)
        print(probability)
        # print(result)
        # # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=probability)
        # return jsonify(result=result, probability=probability.yang)
    return None


@app.route('/real-time', methods=['GET'])
def jump():
    return render_template('face.html')


@app.route('/face', methods=['GET', 'POST'])
def face_rec():
    if request.method == 'POST':

        # print(request.json)  # data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4QAIRXhpZgAA/9sAQwAFAwQEBAM
        # data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEB
        # # Get the image from post request
        # img = base64_to_pil(request.json)

        img = np.asarray(base64_to_pil(request.json))#.convert('L')#.resize((48, 48))
        # img = np.asarray(img).reshape(-1, 48, 48, 1).astype('float32') / 255
        # print(img.shape)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,1)  # 转换为灰度图像，参数CV_RGB2GRAY是RGB到gray。
        # print(img_gray)
        faces = faceCascade.detectMultiScale(  # 人脸检测,使用的是 detectMultiScale函数
            img_gray,  # 待检测图片，一般为灰度图像加快检测速度；
            # scaleFactor=1.1,  # 表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
            # minNeighbors=5,
            # 表示构成检测目标的相邻矩形的最小个数(默认为3个)。如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，这种设定值一般用在用户自定义对检测结果的组合程序上；
            # minSize=(30, 30),  # 用来限制得到的目标区域的范围。
            # flags=cv2.cv.CV_HAAR_SCALE_IMAGE#要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，因此这些区域通常不会是人脸所在区域；
            # flags=0
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            face_image_gray = img_gray[y:y + h, x:x + w]
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 使用对角线的两点pt1，pt2画一个矩形轮廓或者填充矩形
            resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)

            image = resized_img.reshape(1, 1, 48, 48)
            img = np.asarray(image).reshape(-1, 48, 48, 1).astype('float32') / 255
            # list_of_list = model.predict(image, batch_size=1, verbose=1)
            # angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
            p_class_realtime = predict_class(img)
            p_value_realtime = predict_value(img)
            print("预测结果(lable)：", p_class_realtime)
            print("预测结果：", label_dict[p_class_realtime[0]])
            print("每个预测值：", p_value_realtime)

        # 终端显示每个预测结果对应的预测值
        # probability_yang = show_predict_result(p_class1, p_value1)

            # prediction = model.predict_classes(img)
            # prediction_probability = model.predict(img)
            # print(prediction)
            # print(prediction_probability)

            # # # Make prediction
            # preds = model_predict(img, model)
            #
            # # Process your result for human
            # probability = "{:.3f}".format(np.amax(p_class[0]))  # Max probability
            # pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
            #
            result_real_time= str(label_dict[p_class_realtime[0]])  # Convert to string
            result_real_time = result_real_time.replace('_', ' ').capitalize()
            # probability =str(p_value)
            # probability= json.dumps(p_value.item())
            # probability=p_value.tolist()

            # probability = str(p_value)  # Convert to string
            # probability = probability.replace('_', ' ').capitalize()
            probability_real_time= str(",".join(str(float(_ * 100)) for _ in p_value_realtime[0]))

            print(result_real_time)
            print(probability_real_time)
            # # Serialize the result, you can add additional fields
        return jsonify(result=result_real_time, probability=probability_real_time)
        # return jsonify(result=result, probability=probability.yang)
    return None


if __name__ == '__main__':
    app.run()
