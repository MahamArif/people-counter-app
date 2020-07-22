# Project Write-Up

## Models used in the Application

I have tried the following two pre-trained models in this application, from Tensorflow Object Detection Model Zoo:

- [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
- [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

## Commands for Conversion to Intermediate Representation

- ssd_mobilenet_v2_coco:

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

- faster_rcnn_inception_v2_coco:

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```

## Commands for Running the Application on CPU

- ssd_mobilenet_v2_coco:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intermediate_representations/ssd_mobilenet_v2_coco/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

- faster_rcnn_inception_v2_coco:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intermediate_representations/faster_rcnn_inception_v2_coco/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 -n F-RCNN |  ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Explaining Custom Layers

Model Optimizer and the hardware on which inference is performed supports a particular list of layers that are known as supported layers.
Any layer which is not in the list of supported layers is known as a custom layer. 

There are multiple ways to handle custom layers in Model Optimizer, depending on the original model framework. Custom layers can be registered as an extension to the Model Optimizer. Another way is to use the model framework to calculate the output shape for that particular layer.

An extension plugin can be added to the Inference Engine to support additional layers on particular hardware.

I have used the CPU extension to handle custom layers in my application.

Some of the potential reasons for handling custom layers are:

Model Optimizer should support all the layers of the model to make a correct intermediate representation. As model optimizer cannot convert unsupported layers, we need to handle them to make our application perform correctly. Similarly, if the hardware doesn't support all the layers of the model, we need to add extension plugins to add support for additional layers, otherwise, the inference cannot be performed.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:

- To compare inference time before conversion I wrote a script to load the pre-trained model in tensorflow and make inference. After conversion to Intermediate Representation, I calculated inference time using time library in python.

- For pre-trained model before conversion, I used theoretical accuracy explained in papers. After conversion, I calculated accuracy using correct predictions per frames.

The accuracy, inference time and model size for both the models, ssd_mobilenet_v2_coco and faster-rcnn_inception_v2_coco, before and after conversion to Intermediate Representations are given below:

### Pre-conversion Statistics

Model Name | Size | Inference Time | Accuracy
| :----------- | :-------: | :------: | :-------------: | 
SSD | 66.5 MB | 169 ms | 80%
Faster RCNN | 54.5 MB | 3544 ms | 90%

### Post-conversion Statistics

Model Name | Size | Inference Time | Accuracy
| :----------- | :-------: | :------: | :-------------: | 
SSD | 64.2 MB | 68 ms | 75%
Faster RCNN | 50.8 MB | 935 ms | 90%

The CPU overhead was greater before conversion as the model was not optimized, but after conversion to Intermediate Representation, the model performs really well in terms of CPU overhead and resources.

If the model is deployed at the edge, there is minimal network cost involved if we need to send potential frames to the cloud.
However, if a model is deployed on the cloud, real-time processing will not be possible because of the network latency. This communication will also be expensive as it consumes power and bandwidth.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
- People counter app can be used in gaming zones, to identify how many people are visiting the gaming area daily.
- This application can also be utilized in Bank ATM's to determine the average amount of cash needed regularly.
- People's counter app, along with a recognition feature, can be used in libraries to identify potential regular readers.
- This application can also be used in shopping malls and stores, to determine their regular customers and the duration for which they remain in the mall.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

- Lightning

  Proper lightning in the environment will be needed by the model to perform correctly. Dim light may result in poor performance and incorrect predictions by the model.

- Model Accuracy

  Model should have good accuracy (above 90%) in the case where exact count and duration are needed. A model with low accuracy will result in an incorrect count. However, if the exact count is not crucial for the use case, we can use a model with a low accuracy but higher speed.

- Camera Focal Length

  Camera focal length should be chosen according to the area of the environment to observe. 

- Image Size
  
  Low image size/resolution may result in incorrect predictions by the model. It would be ideal if images of the same size/resolution are supplied as input to the model on which it is trained.

## Conclusion

For this application, I investigated two models, SSD Mobilenet V2 and Faster RCNN Inception V2. SSD is much faster but its performance is not that good which results in incorrect total count and duration. On the other hand, the performance of Faster RCNN was great in terms of accuracy, but it is very slow as compared to SSD. This can be a drawback when talking about an edge application where speed is equally important.
