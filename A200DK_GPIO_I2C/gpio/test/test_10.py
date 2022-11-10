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

import time
import a200dkgpio as gpio

# GPIO4 INPUT测试 & GPIO5 OUTPUT测试
# GPIO4直连GPIO5

gpio.gpio_init()
print("setup GPIO4 INPUT")
gpio.setup(gpio.GPIO4, gpio.INPUT)
print("setup GPIO5 OUTPUT")
gpio.setup(gpio.GPIO5, gpio.OUTPUT)
print("output GPIO5 LOW")
gpio.output(gpio.GPIO5, gpio.LOW)
res = gpio.gpio_function(gpio.GPIO5)
print("GPIO5 function: ", res)
res = gpio.gpio_function(gpio.GPIO4)
print("GPIO4 function: ", res)
res = gpio.input(gpio.GPIO4)
print("GPIO4 input: ", res)
print("sleep")
time.sleep(1)
print("output GPIO5 HIGH")
gpio.output(gpio.GPIO5, gpio.HIGH)
res = gpio.input(gpio.GPIO4)
print("GPIO4 input: ", res)
print("sleep")
time.sleep(1)
gpio.gpio_close()