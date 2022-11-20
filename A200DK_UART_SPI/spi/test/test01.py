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
import spidev


spi = spidev.SpiDev()
spi.open(0, 0)

SPI_SPEED = 1000000
spi.max_speed_hz = SPI_SPEED
spi.mode = 0b01
spi.writebytes2([01, 01])

spi.close()
