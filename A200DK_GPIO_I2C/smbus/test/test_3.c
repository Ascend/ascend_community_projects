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
#include <smbus.h>

int main(void)
{
    i2c_2_init();
    int res;
    res = get_pec();
    printf("pec %d\n", res);
    enable_pec(1);
    res = get_pec();
    printf("pec %d\n", res);
    res=read_byte(0x68);
    printf("read_byte(0x68): %#x\n", res);
    i2c_2_close();
    return 0;
}
