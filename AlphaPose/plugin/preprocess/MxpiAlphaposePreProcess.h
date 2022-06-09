
#ifndef ALPHAPOSEPREPROCESS_MXPIALPHAPOSEPREPROCESS_H
#define ALPHAPOSEPREPROCESS_MXPIALPHAPOSEPREPROCESS_H
#include "opencv2/opencv.hpp"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"


/**
 * @api
 * @brief Definition of MxpiAlphaposePreProcess class.
 */

namespace MxPlugins {
    class MxpiAlphaposePreProcess : public MxTools::MxPluginBase {
    public:
        /**
         * @brief Initialize configure parameter.
         * @param configParamMap
         * @return APP_ERROR
         */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

        /**
         * @brief DeInitialize configure parameter.
         * @return APP_ERROR
         */
        APP_ERROR DeInit() override;

        /**
         * @brief Process the data of MxpiBuffer.
         * @param mxpiBuffer
         * @return APP_ERROR
         */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

        /**
         * @brief Definition the parameter of configure properties.
         * @return std::vector<std::shared_ptr<void>>
         */
        static std::vector<std::shared_ptr<void>> DefineProperties();

        /**
         * @brief Overall process to generate pretreated images
         * @param srcMxpiObjectList - Source MxpiObjectList containing object data about input image
         * @param srcMxpiVisionList - Source MxpiTensorPackageList containing input image
         * @param dstMxpiVisionList - Target MxpiVisionList containing detection result list
         * @return APP_ERROR
         */
        APP_ERROR GenerateVisionList(const MxTools::MxpiObjectList &srcMxpiObjectList,
                                     const MxTools::MxpiVisionList &srcMxpiVisionList,
                                     MxTools::MxpiVisionList &dstMxpiVisionList);
        /**
         * @brief Prepare output in the format of MxpiVisionList
         * @param affinedImages - The image after affine transformation
         * @param dstMxpiVisionList - Target data in the format of MxpiVisionList
         * @return APP_ERROR
         */
        APP_ERROR GenerateMxpiOutput(std::vector<cv::Mat> &affinedImages,
                                     MxTools::MxpiVisionList &dstMxpiVisionList);

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer &buffer, const std::string plugin_name,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::string imageDecoderName_;
        std::ostringstream ErrorInfo_;
    };
}
#endif // ALPHAPOSEPREPROCESS_MXPIALPHAPOSEPREPROCESS_H