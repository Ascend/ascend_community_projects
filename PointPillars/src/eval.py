"""
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
import numpy as np
import fire


def evaluate(file_dir="../result/test/", benchmark_dir="../benchmark/test/"):
    om = np.fromfile(f"{file_dir}/result.bin", dtype=np.float32).reshape(-1, 7)
    benchmark = np.fromfile(f"{benchmark_dir}/result.bin", dtype=np.float32).reshape(-1, 7)
    cnt = om.shape
    error = 0
    for i in range(cnt[0]):
        miss = 0
        benchmark_sum = 0
        for j in range(0, 3):
            miss += abs(benchmark[i][j] - om[i][j])
            benchmark_sum += abs(benchmark[i][j])

        error = max(error, miss / benchmark_sum)
   
    print('the error of the model is :', error)


if __name__ == '__main__':
    fire.Fire()
