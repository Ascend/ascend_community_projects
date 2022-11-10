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

import a200dkgpio as gpio

# GPIO1 输入测试
# GPIO1接+3.3V或GND

GPIO_ID = 1
gpio.gpio_init()
print("setup gpio%d INTPUT" % GPIO_ID)
gpio.setup(GPIO_ID, gpio.INPUT)
res = gpio.input(GPIO_ID)
print("gpio%d input: %d" % (GPIO_ID, res))
res = gpio.gpio_function(GPIO_ID)
print("gpio%d function: %d" % (GPIO_ID, res))
gpio.gpio_close()