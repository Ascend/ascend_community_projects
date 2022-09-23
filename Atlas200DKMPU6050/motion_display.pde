/**
* Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

import processing.serial.*;

Serial myPort;
String data = "";
float pitch, yaw, roll;
int baudrate = 9600;
int box_length = 200, box_height = 50, box_width = 150;

void setup(){
  size(800, 480, P3D);
  myPort = new Serial(this, "COM3", baudrate);
}

void draw(){
  background(255);
  fill(255, 0, 0);
  directionalLight(255, 255, 255, 1, 1, -1);
  pushMatrix();
  translate(width / 2, height / 2);
  rotateX(radians(roll));
  rotateY(radians(yaw));
  rotateZ(radians(-pitch));
  
  box(box_length, box_height, box_width);
  popMatrix();
}

void serialEvent (Serial myPort) { 
  data = myPort.readStringUntil('\n');
  if (data != null) {
    data = trim(data);
    String items[] = split(data, ',');
    if (items.length > 1) {
      pitch = float(items[0]);
      yaw = float(items[1]);
      roll = float(items[2]);
    }
  }
}
