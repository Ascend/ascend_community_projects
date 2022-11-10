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
* GPIO3 输出高电平测试
* 测试外设：发光二极管
* GPIO3接发光二极管正极，发光二极管负极接地
*/

int main(void)
{
    int res;
    int gpio = 3;
    int sleep_time = 2;
    gpio_init();
    printf("setup gpio%d OUTPUT\n", gpio);
    setup(gpio, OUTPUT);
    printf("output gpio%d HIGH\n", gpio);
    output(gpio, HIGH);
    res = gpio_function(gpio);
    printf("gpio%d function: %d\n", gpio, res);
    res = input(gpio);
    printf("gpio%d value: %d\n", gpio, res);
	sleep(sleep_time);
    gpio_close();
    return 0;
}