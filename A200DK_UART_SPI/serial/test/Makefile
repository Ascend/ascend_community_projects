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
#DEBUG = y

# Add your debugging flag (or not) to CFLAGS
ifeq ($(DEBUG),y)
  DEBFLAGS = -O -g
# "-O" is needed to expand inlines
else
  DEBFLAGS = -O2
endif

CROSS_COMPILE	?= aarch64-linux-gnu-
CC=$(CROSS_COMPILE)gcc
CFLAGS= -Wall

CFLAGS += $(DEBFLAGS)
LDFLAGS = 

cname?=serial_test
oname=$(cname).o
targetname = $(cname)
all: $(targetname)

#compiler ruler
$(targetname):$(oname)  c_serial.o 
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)
	rm  *.o  
	rm -f *.bak
%.o: %.c
	$(CC) $(CFLAGS) -c $<
clean:
	rm -f targetname 
