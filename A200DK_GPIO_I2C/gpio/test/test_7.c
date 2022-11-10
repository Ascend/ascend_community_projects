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
#include <gpio.h>

/*
* GPIO1 输入测试
* GPIO1接+3.3V或GND
*/
int main(void)
{
    int res;
    int gpio = 1;
    gpio_init();
    printf("setup gpio%d INTPUT\n", gpio);
    setup(gpio, INPUT);
    res = input(gpio);
    printf("gpio%d input: %d\n", gpio, res);
    res = gpio_function(gpio);
    printf("gpio%d function: %d\n", gpio, res);
    gpio_close();
    return 0;
}