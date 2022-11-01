"""
# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from torch.autograd import Variable
import torch.onnx
from models import FireClassifier
cpu = torch.device("cpu")
net = FireClassifier(backbone='densenet121', pretrained=False)
dummy_input = Variable(torch.randn(1, 3, 224, 224)).to(cpu)
INPUT_MODEL_PATH = './models/firedetect-densenet121-pretrained.pt'
OUTPUT_MODEL_PATH = './models/densenet.onnx'
state_dict = torch.load(INPUT_MODEL_PATH, map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
torch.onnx.export(net, dummy_input, OUTPUT_MODEL_PATH, 
                  input_names=["image"], output_names=["output"], verbose=True)