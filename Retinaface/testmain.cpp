/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include <RetinafaceDetection.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <string>
#include <vector>

#include "MxBase/Log/Log.h"
std::string imgFile;
void InitRetinafaceParam(InitParam& initParam) {
  initParam.deviceId = 0;
  initParam.checkTensor = true;
  initParam.modelPath = "/home/dongyu1/Retinaface0/model/newRetinaface.om";
  initParam.classNum = 1;
  initParam.labelPath = "";
  initParam.ImagePath = "";
}

void GetFileNames(std::string path, std::vector<std::string>& filenames) {
  DIR* pDir;
  struct dirent* ptr;
  if (!(pDir = opendir(path.c_str()))) {
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while ((ptr = readdir(pDir)) != 0) {
    if (std::strcmp(ptr->d_name, ".") != 0 &&
        std::strcmp(ptr->d_name, "..") != 0) {
      filenames.push_back(path + "/" + ptr->d_name);
    }
  }
  closedir(pDir);
}
int main(int argc, char* argv[]) {
  if (argc <= 1) {
    LogWarn << "Please input image path, such as './RetinafacePostProcess "
               "test.jpg'.";
    return APP_ERR_OK;
  }
  imgFile = argv[1];

  InitParam initParam;
  InitRetinafaceParam(initParam);
  auto Retinaface = std::make_shared<RetinafaceDetection>();

  APP_ERROR ret = Retinaface->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "RetinafaceDetection init failed, ret=" << ret << ".";
    return ret;
  }

  std::vector<std::string> ImageFile;
  GetFileNames(imgFile, ImageFile);
  int cnt = 0;
  for (uint32_t i = 0; i < ImageFile.size(); i++) {
    std::vector<std::string> imagePath;
    std::cout << "Image File = " << ImageFile[i] << std::endl;
    GetFileNames(ImageFile[i], imagePath);
    for (uint32_t j = 0; j < imagePath.size(); j++) {
      Retinaface->Process(imagePath[j]);
      std::cout << "Image Name = " << imagePath[j] << "\n";
      cnt++;
      std::cout << "picture number is " << cnt << std::endl;
    }
  }
  Retinaface->DeInit();
  return APP_ERR_OK;
}