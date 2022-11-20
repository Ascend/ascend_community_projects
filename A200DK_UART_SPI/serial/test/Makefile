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
