#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.model_name = None
        self.net_plugin = None
        self.input_blob = None
        self.output_blob = None
        self.infer_request_handle = None

    def load_model(self, model, model_name, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
            
        ### Initialize the plugin ###
        self.plugin = IECore()
        
        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.model_name = model_name
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### Check for supported layers ###
        ### Add any necessary extensions ###
        unsupported_layers = self.get_unsupported_layers(device)
        if len(unsupported_layers):
            print("Unsupported layers found: {}".format(unsupported_layers))
            if cpu_extension and "CPU" in device:
                print("Using CPU Extension")
                self.plugin.add_extension(cpu_extension, device)
                unsupported_layers = self.get_unsupported_layers(device)
                if len(unsupported_layers):
                    print("Unsupported layers found: {}".format(unsupported_layers))
                    print("Check whether extensions are available to add to IECore.")
                    sys.exit(1)
            else:
                print("Check whether extensions are available to add to IECore.")
                sys.exit(1)
            
        ### Return the loaded inference plugin ###
        self.net_plugin = self.plugin.load_network(self.network, device)
        
        ### Get the input layer ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return self.net_plugin

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        if self.model_name == "F-RCNN":
            return self.network.inputs['image_tensor'].shape
        else:
            return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        input_data = None
        if self.model_name == "F-RCNN":
            input_data = {'image_tensor': image, 'image_info': image.shape[1:]}
        else:
            input_data = {self.input_blob: image}
        self.infer_request_handle = self.net_plugin.start_async(request_id=0, 
            inputs=input_data)
        return self.infer_request_handle

    def wait(self):
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        status = self.infer_request_handle.wait(-1)
        return status

    def get_output(self):
        ### Extract and return the output results ###
        return self.infer_request_handle.outputs[self.output_blob]
    
    def get_unsupported_layers(self, device):
        ### Return the layers not supported by the network ###
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        return unsupported_layers
    
