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
SPI_SPEED_2 = 2000000
BITS_PER_WORD_32 = 32
spi.max_speed_hz = SPI_SPEED
spi.mode = 0b01
print(spi.xfer([1, 1]))

print(spi.mode)
print(spi.bits_per_word)
print(spi.max_speed_hz)

spi.mode = 0
spi.bits_per_word = BITS_PER_WORD_32
spi.max_speed_hz = SPI_SPEED_2  

print(spi.mode)
print(spi.bits_per_word)
print(spi.max_speed_hz)

spi.close()
