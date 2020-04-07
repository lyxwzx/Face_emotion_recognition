var predResult = document.getElementById("pred-result");
var predValue  = document.getElementById("pred-value");

// 获取摄像头
window.addEventListener("DOMContentLoaded", function () {
        var video = document.getElementById("video"), canvas, context;
        try {
            canvas = document.createElement("canvas");
            context = canvas.getContext("2d");
        } catch (e) { alert("not support canvas!"); return; }
        //提醒用户需要使用音频、视频输入设备
        navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
        if (navigator.getUserMedia)
            navigator.getUserMedia(
                { "video": true },
                function (stream) {
					 alert("允许调用摄像头！！！");
                    if (video.mozSrcObject !== undefined){
							video.mozSrcObject = stream;
					}
                    else{
						//video.src = ((window.URL || window.webkitURL || window.mozURL || window.msURL)  && HTMLMediaElement.srcObject(stream)) || stream ;
						video.srcObject=stream;
						video.play();
					}
                },
                function (error) {
                    if(error.PERMISSION_DENIED)console.log("用户拒绝了浏览器请求媒体的权限",error.code);
                    if(error.NOT_SUPPORTED_ERROR)console.log("当前浏览器不支持拍照功能",error.code);
                    if(error.MANDATORY_UNSATISFIED_ERROR)console.log("指定的媒体类型未接收到媒体流",error.code);
                    alert("Video capture error: " + error.code +error);
                }
            );
        else alert("Native device media streaming (getUserMedia) not supported in this browser");

        setInterval(function () {
            //方法会不停地调用函数，直到 clearInterval() 被调用或窗口被关闭。
             //context.drawImage(video, 0, 0, canvas.width = video.videoWidth, canvas.height = video.videoHeight);
              //image=canvas.toDataURL("image/png", 1.0)//.substr(22)  方法返回一个包含图片展示的 data URI 。可以使用 type 参数其类型，默认为 PNG 格式。图片的分辨率为96dpi
              canvas.getContext("2d").drawImage(video, 0, 0, canvas.width = video.videoWidth, canvas.height = video.videoHeight);//将图片绘制到canvas中
              dataURL=canvas.toDataURL('image/jpeg'); //转换图片为dataURL
                fetch("/face", {
                  method: "POST",
                  headers: {
                      "Content-Type": "application/json"
                  },
                  body: JSON.stringify(dataURL)
                })
                .then(resp => {
                      if (resp.ok)
                        resp.json().then(data => {
                            displayResult(data);
                            // displayValue(data);
                      });
                  })
                .catch(err => {
                      console.log("An error occured", err.message);
                      window.alert("Oops! Something went wrong.");
                  });
            // $.post('/face', { "img": canvas.toDataURL().substr(22) }, function (data, status) {
            //     if (status == "success" && data != "no")location.href = "/Users/liyixin/PycharmProjects/Fer2013lyx/Result" + data;
            // }, "text");
        }, 1000);
    }, false);

function displayResult(data) {
    var label = ["Angry","Disgust","Fear","Happy","Sad","Suprise","Neutral"];
    var strr = "预测结果为："+data.result+"<br/>";
    var value = new Array(); //定义一数组
    value = data.probability.split(","); //字符分割
    for (i=0; i<value.length; i++){
 　  　strr += label[i]+" probability： "+value[i]+"<br/>"; //分割后的字符输出
}
    predResult.innerHTML = strr;
    show(predResult);
}

//
// function displayValue(data) {
//     var label = ["Angry","Disgust","Fear","Happy","Sad","Suprise","Neutral"];
//     var strr ="";
//     var value = new Array(); //定义一数组
//     value = data.probability.split(","); //字符分割
//     for (i=0; i<value.length; i++){
//  　  　strr += label[i]+" probability： "+value[i]+"<br/>"; //分割后的字符输出
// }
//   predValue.innerHTML=strr;
//   show(predValue);
// }