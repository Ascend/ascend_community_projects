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

import a200dksmbus as i2c

i2c.i2c_2_init()
READ_LENGTH = 16
ADDRESS = 0x68
REG = 0x00
buf = i2c.read_i2c_block_data(ADDRESS, REG, READ_LENGTH)
print(buf)
i2c.i2c_2_close()
