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

#ifndef CRP_H
#define CRP_H

#include "MxBase/E2eInfer/Image/Image.h"
#include "MxBase/E2eInfer/Rect/Rect.h"
#include "MxBase/E2eInfer/Size/Size.h"

#include "acl/dvpp/hi_dvpp.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"

#define CONVER_TO_PODD(NUM) (((NUM) % 2 != 0) ? (NUM) : ((NUM)-1))
#define CONVER_TO_EVEN(NUM) (((NUM) % 2 == 0) ? (NUM) : ((NUM)-1))
#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

MxBase::Rect GetPasteRect(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight)
{
    bool widthRatioLarger = true;
    if(outputWidth == 0)
    {
        LogError << "outputWidth equals to 0.";
    }
    if(outputHeight == 0)
    {
        LogError << "outputHeight equals to 0.";
    }
    float resizeRatio = static_cast<float>(inputWidth) / outputWidth;
    if (resizeRatio < (static_cast<float>(inputHeight) / outputHeight))
    {
        resizeRatio = static_cast<float>(inputHeight) / outputHeight;
        widthRatioLarger = false;
    }
    
    // (x0, y0)是左上角坐标，(x1, y1)是右下角坐标，采用图片坐标系
    uint32_t x0;
    uint32_t y0;
    uint32_t x1;
    uint32_t y1;
    if (widthRatioLarger)
    {
        // 原图width大于height
        x0 = 0;
        y0 = 0;
        x1 = outputWidth-1;
        y1 = inputHeight / resizeRatio - 1;
    }
    else
    {
        // 原图height大于width
        x0  = 0;
        y0 = 0;
        x1 = inputWidth / resizeRatio - 1;
        y1 = outputHeight - 1;
    }
    x0 = DVPP_ALIGN_UP(CONVER_TO_EVEN(x0), 16); // 16对齐
    x1 = DVPP_ALIGN_UP((x1 - x0 + 1), 16) + x0 - 1;
    y0 = CONVER_TO_EVEN(y0);
    y1 = CONVER_TO_PODD(y1);
    MxBase::Rect res(x0, y0, x1, y1);
    return res;
}

MxBase::Image ConstructImage(uint32_t resizeWidth, uint32_t resizeHeight)
{
    void *addr;
    uint32_t dataSize = resizeWidth * resizeHeight * 3 / 2;
    auto ret = hi_mpi_dvpp_malloc(0, &addr, dataSize);
    if (ret != APP_ERR_OK)
    {
        LogError << "hi_mpi_dvpp_malloc fail :" << ret;
    }
    // 第三个参数从128改成了0
    ret = aclrtMemset(addr, dataSize, 0, dataSize);
    if (ret != APP_ERR_OK)
    {
        LogError << "aclrtMemset fail :" << ret;
    }
    std::shared_ptr<uint8_t> data((uint8_t *)addr, hi_mpi_dvpp_free);
    MxBase::Size imageSize(resizeWidth, resizeHeight);
    MxBase::Image pastedImg(data, dataSize, 0, imageSize);
    return pastedImg;
}

std::pair<MxBase::Rect, MxBase::Rect> GenerateRect(uint32_t originalWidth, uint32_t originalHeight, uint32_t resizeWidth, uint32_t resizeHeight)
{
    uint32_t x1 = CONVER_TO_PODD(originalWidth - 1);
    uint32_t y1 = CONVER_TO_PODD(originalHeight - 1);
    MxBase::Rect cropRect(0, 0, x1, y1);
    MxBase::Rect pasteRect = GetPasteRect(originalWidth, originalHeight, resizeWidth, resizeHeight);
    std::pair<MxBase::Rect, MxBase::Rect> cropPasteRect(cropRect, pasteRect);
    return cropPasteRect;
}

MxBase::Image resizeKeepAspectRatioFit(uint32_t originalWidth, uint32_t originalHeight, uint32_t resizeWidth, uint32_t resizeHeight, MxBase::Image &decodeImage, MxBase::ImageProcessor& imageProcessor)
{
    std::pair<MxBase::Rect, MxBase::Rect> cropPasteRect = GenerateRect(originalWidth, originalHeight, resizeWidth, resizeHeight);
    MxBase::Image resizeImage = ConstructImage(resizeWidth, resizeHeight);
    auto ret = imageProcessor.CropAndPaste(decodeImage, cropPasteRect, resizeImage);
    if (ret != APP_ERR_OK)
    {
        LogError << "CropAndPaste fail :" << ret;
    }
    return resizeImage;
}

#endif // CRP_H