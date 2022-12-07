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
import a200dkserial

BAUDRATE_DEFAULT = 115200
TIMEOUT_MS = 2000
DATABITS_7 = 7
STOPBITS_2 = 2
READ_LEN = 10
WAITING = 3


ser = a200dkserial.Serial(1, BAUDRATE_DEFAULT)  # open serial port

print(f"波特率： {ser.baudrate}")
print(f"数据位： {ser.databits}")
print(f"奇偶校验： {ser.parity}")
print(f"xonxoff： {ser.xonxoff}")
print(f"rtscts: {ser.rtscts}")
print(f"vtime: {ser.vtime}")
print(f"vim: {ser.vmin}")
print(f"停止位: {ser.stopbits}")
ser.baudrate = BAUDRATE_DEFAULT
ser.databits = DATABITS_7
ser.parity = 1
ser.xonxoff = False
ser.rtscts = False
ser.stopbits = STOPBITS_2
print("请输入：")
a = ser.read(READ_LEN, TIMEOUT_MS)
if a:
    print(f"输入： {a}")

ser.write(b"hello\r\n")
ser.input_waiting(WAITING)
ser.output_waiting(WAITING)
ser.poll(READ_LEN)
print(f"波特率： {ser.baudrate}")
print(f"数据位： {ser.databits}")
print(f"奇偶校验： {ser.parity}")
print(f"xonxoff： {ser.xonxoff}")
print(f"rtscts: {ser.rtscts}")
print(f"vtime: {ser.vtime}")
print(f"vim: {ser.vmin}")
print(f"停止位: {ser.stopbits}")
print(str(ser))
ser.close()             # close port
