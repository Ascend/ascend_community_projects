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

# GPIO6 cleanup测试
# 测试外设：发光二极管
# GPIO6接发光二极管正极，发光二极管负极接地

GPIO_ID = 6
gpio.gpio_init();
print("setup gpio%d OUTPUT" % GPIO_ID)
gpio.setup(GPIO_ID, gpio.OUTPUT)
print("output gpio%d HIGH" % GPIO_ID)
gpio.output(GPIO_ID, gpio.HIGH)
res = gpio.gpio_function(GPIO_ID)
print("gpio%d function: %d" % (GPIO_ID, res))
print("sleep")
time.sleep(1)
print("setup gpio%d OUTPUT" % GPIO_ID)
gpio.setup(GPIO_ID, gpio.OUTPUT)
print("cleanup gpio%d" % GPIO_ID)
gpio.cleanup(GPIO_ID)
res = gpio.gpio_function(GPIO_ID)
print("gpio%d function: %d" % (GPIO_ID, res))
print("setup gpio%d OUTPUT", GPIO_ID)
gpio.setup(GPIO_ID, gpio.OUTPUT)
res = gpio.gpio_function(GPIO_ID)
print("gpio%d function: %d" % (GPIO_ID, res))
gpio.gpio_close()
