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

#include "Hungarian.h"
#include <cstring>
#include <sys/time.h>
#include "MxBase/Log/Log.h"

namespace {
    const int INF = 0x3f3f3f3f;
    const int VISITED = 1;
    const int HUNGARIAN_CONTENT = 7;
    const int X_MATCH_OFFSET = 0;
    const int Y_MATCH_OFFSET = 1;
    const int X_VALUE_OFFSET = 2;
    const int Y_VALUE_OFFSET = 3;
    const int SLACK_OFFSET = 4;
    const int X_VISIT_OFFSET = 5;
    const int Y_VISIT_OFFSET = 6;
}

APP_ERROR HungarianHandleInit(HungarianHandle &handle, int row, int cols)
{
    handle.max = (row > cols) ? row : cols;
    auto adjMat = std::shared_ptr<int>();
    adjMat.reset(new int[handle.max * handle.max], std::default_delete<int[]>());
    if (adjMat == nullptr) {
        LogFatal << "HungarianHandleInit new failed";
        return APP_ERR_ACL_FAILURE;
    }

    handle.adjMat = adjMat;

    void* ptr[HUNGARIAN_CONTENT] = {nullptr};
    for (int i = 0; i < HUNGARIAN_CONTENT; ++i) {
        ptr[i] = malloc(handle.max * sizeof(int));
        if (ptr[i] == nullptr) {
            LogFatal << "HungarianHandleInit Malloc failed";
            return APP_ERR_ACL_FAILURE;
        }
    }

    handle.xMatch.reset((int *)ptr[X_MATCH_OFFSET], free);
    handle.yMatch.reset((int *)ptr[Y_MATCH_OFFSET], free);
    handle.xValue.reset((int *)ptr[X_VALUE_OFFSET], free);
    handle.yValue.reset((int *)ptr[Y_VALUE_OFFSET], free);
    handle.slack.reset((int *)ptr[SLACK_OFFSET], free);
    handle.xVisit.reset((int *)ptr[X_VISIT_OFFSET], free);
    handle.yVisit.reset((int *)ptr[Y_VISIT_OFFSET], free);
    return APP_ERR_OK;
}

static void HungarianInit(HungarianHandle &handle, const std::vector<std::vector<int>> &cost, int rows, int cols)
{
    int i, j, value;
    if (rows > cols) {
        handle.transpose = true;
        handle.cols = rows;
        handle.rows = cols;
        handle.resX = handle.yMatch.get();
        handle.resY = handle.xMatch.get();
    } else {
        handle.transpose = false;
        handle.rows = rows;
        handle.cols = cols;
        handle.resX = handle.xMatch.get();
        handle.resY = handle.yMatch.get();
    }

    for (i = 0; i < handle.rows; ++i) {
        handle.xValue.get()[i] = 0;
        handle.xMatch.get()[i] = -1;
        for (j = 0; j < handle.cols; ++j) {
            if (handle.transpose) {
                value = cost[j][i];
            } else {
                value = cost[i][j];
            }
            handle.adjMat.get()[i * handle.cols + j] = value;
            if (handle.xValue.get()[i] < value) {
                handle.xValue.get()[i] = value;
            }
        }
    }

    for (i = 0; i < handle.cols; ++i) {
        handle.yValue.get()[i] = 0;
        handle.yMatch.get()[i] = -1;
    }
}

static bool Match(HungarianHandle &handle, int id)
{
    int j, delta;
    handle.xVisit.get()[id] = VISITED;
    for (j = 0; j < handle.cols; ++j) {
        if (handle.yVisit.get()[j] == VISITED) {
            continue;
        }
        delta = handle.xValue.get()[id] + handle.yValue.get()[j] - handle.adjMat.get()[id * handle.cols + j];
        if (delta == 0) {
            handle.yVisit.get()[j] = VISITED;
            if (handle.yMatch.get()[j] == -1 || Match(handle, handle.yMatch.get()[j])) {
                handle.yMatch.get()[j] = id;
                handle.xMatch.get()[id] = j;
                return true;
            }
        } else if (delta < handle.slack.get()[j]) {
            handle.slack.get()[j] = delta;
        }
    }
    return false;
}

int HungarianSolve(HungarianHandle &handle, const std::vector<std::vector<int>> &cost, int rows, int cols)
{
    HungarianInit(handle, cost, rows, cols);
    int i, j, delta;
    for (i = 0; i < handle.rows; ++i) {
        while (true) {
            std::fill(handle.xVisit.get(), handle.xVisit.get() + handle.rows, 0);
            std::fill(handle.yVisit.get(), handle.yVisit.get() + handle.cols, 0);
            for (j = 0; j < handle.cols; ++j) {
                handle.slack.get()[j] = INF;
            }
            if (Match(handle, i)) {
                break;
            }
            delta = INF;
            for (j = 0; j < handle.cols; ++j) {
                if (handle.yVisit.get()[j] != VISITED && delta > handle.slack.get()[j]) {
                    delta = handle.slack.get()[j];
                }
            }
            if (delta == INF) {
                LogDebug << "Hungarian solve is invalid!";
                return -1;
            }
            for (j = 0; j < handle.rows; ++j) {
                if (handle.xVisit.get()[j] == VISITED) {
                    handle.xValue.get()[j] -= delta;
                }
            }
            for (j = 0; j < handle.cols; ++j) {
                if (handle.yVisit.get()[j] == VISITED) {
                    handle.yValue.get()[j] += delta;
                }
            }
        }
    }
    return handle.rows;
}
