"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import mimetypes
from enum import Enum

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

MAX_FRAME_COUNT_FOR_CONFIDENCE = 4

class FileTypes(Enum):
    CAM = 'CAM'
    IMAGE = 'IMAGE'
    VIDEO = 'VIDEO'

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-n", "--model_name", type=str, default="SSD",
                        help="Model name used for person detection"
                        "(SSD by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    ### Initialise the class ###
    infer_network = Network()
    
    ### Set Probability threshold for detections ###
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.model_name, args.device, args.cpu_extension)

    ### Handle the input stream ###
    image_flag = False
    input_shape = infer_network.get_input_shape()
    input_file_type = get_input_file_type(args.input)
    
    if input_file_type == None:
        print("Unrecognized file format")
        sys.exit(1)
        
    if input_file_type == FileTypes.CAM.value:
        args.input = 0
    elif input_file_type == FileTypes.IMAGE.value:
        image_flag = True
        
    ### Get and open video capture ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    ### Grab the shape of the input ###
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    ### Variables ###
    total_count = 0
    last_count = 0
    start_time = 0
    contiguous_frames_count = 0
    
    ### Loop until stream is over ###
    while cap.isOpened():
        
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### Break if escape key pressed ###
        if key_pressed == 27:
            break

        ### Pre-process the image as needed ###
        processed_frame = preprocessing(frame, input_shape[2], input_shape[3])

        ### Start asynchronous inference for specified request ###
        inference_start_time = time.time()
        infer_network.exec_net(processed_frame)
        
        ### Wait for the result ###
        if infer_network.wait() == 0:
            
            ### Get the results of the inference request ###
            result = infer_network.get_output()
            
            ### Extract any desired stats from the results ###
            output_frame, current_count = draw_bounding_box(frame, result, width, height, prob_threshold)
            
            ### Print the inference time ###
            inference_duration = time.time() - inference_start_time
            inference_time_message = "Inference time: {:.3f}ms"\
                               .format(inference_duration * 1000)
            cv2.putText(output_frame, inference_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count > last_count or current_count < last_count:
                contiguous_frames_count = 1
            elif current_count > 0 and contiguous_frames_count == MAX_FRAME_COUNT_FOR_CONFIDENCE:
                start_time = time.time()
                total_count = total_count + current_count
                client.publish("person", json.dumps({"total": total_count}))
                contiguous_frames_count = 0
            elif current_count == 0 and contiguous_frames_count == MAX_FRAME_COUNT_FOR_CONFIDENCE:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))
                contiguous_frames_count = 0
            elif contiguous_frames_count > 0:
                contiguous_frames_count = contiguous_frames_count + 1
                
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(output_frame)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite("output_image.jpg", output_frame)
        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def get_input_file_type(input_file):
    """
    Parse and return the input file type.
    """
    if input_file.upper() == FileTypes.CAM.value:
        return FileTypes.CAM.value
    else:
        mime_type = mimetypes.guess_type('./image.png')[0].split('/')[0].upper()
        if FileTypes[mime_type]:
            return FileTypes[mime_type].value
        return None
    
def preprocessing(input_image, height, width):
    """
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    """
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)
    
    return image

def draw_bounding_box(image, result, width, height, threshold):
    """
    Given an input image, result, height and width:
    - Draw bounding box on detected persons
    - Return people count and image with bounding box
    """
    current_count = 0
    for bounding_box in result[0][0]:
        confidence = bounding_box[2]
        if confidence >= threshold:
            current_count = current_count + 1
            xmin = int(bounding_box[3] * width)
            ymin = int(bounding_box[4] * height)
            xmax = int(bounding_box[5] * width)
            ymax = int(bounding_box[6] * height)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
    
    return image, current_count
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
