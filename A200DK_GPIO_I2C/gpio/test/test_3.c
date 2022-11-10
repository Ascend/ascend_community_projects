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
* GPIO4 INPUT测试 & GPIO5 OUTPUT测试
* GPIO4直连GPIO5
*/

int main(void)
{
    int res;
    gpio_init();
    printf("setup GPIO4 INPUT\n");
    setup(GPIO4, INPUT);
    printf("setup GPIO5 OUTPUT\n");
    setup(GPIO5, OUTPUT);
    printf("output GPIO5 LOW\n");
    output(GPIO5, LOW);
    res = gpio_function(GPIO5);
    printf("GPIO5 function: %d\n", res);
    res = gpio_function(GPIO4);
    printf("GPIO4 function: %d\n", res);
    res = input(GPIO4);
    printf("GPIO4 input: %d\n", res);
    printf("sleep\n");
    sleep(1);
    printf("output GPIO5 HIGH\n");
    output(GPIO5, HIGH);
    res = input(GPIO4);
    printf("GPIO4 input: %d\n", res);
    printf("sleep\n");
    sleep(1);
    gpio_close();
    return 0;
}