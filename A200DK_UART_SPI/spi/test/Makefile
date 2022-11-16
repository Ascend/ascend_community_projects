#
#	Description:
#		Makefile for 2.6 device-drivers tests		
#
#	Copyright (C), 2005-2010, Embedded System of China Tech. Co., Ltd.
#
#	Author: Casiawu(wujh@emsyschina.com)

# Comment/uncomment the following line to disable/enable debugging
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

all: spitest

#compiler ruler
spitest:test01.o  c_spidev.o 
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)
	rm  *.o  
	rm -f *.bak
%.o: %.c
	$(CC) $(CFLAGS) -c $<
clean:
	rm -f spitest 
