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

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <smbus.h>

int main(void)
{
    i2c_2_init();
    int read_length = 16;
    int address = 0x68;
    int reg = 0x00;
    unsigned char *buf = read_i2c_block_data(address, reg, read_length);
    printf("read_i2c_block_data(0x68, 0x00, 16): ");
    for (int i = 0; i < read_length; i++)
    {
        printf("%#x ", buf[i]);
    }
    printf("\n");
    i2c_2_close();
    return 0;
}
