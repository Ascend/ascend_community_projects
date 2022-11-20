#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/200dk_serial.h"

int main(void) {
    int fd = 0;
    uint8_t s[] = "Hello World!";
    uint8_t buf[128];
    int ret;

    /* Open /dev/ttyAMAX1 with baudrate 115200, and defaults of 8N1, no flow control */
    if (serial_open(&fd, "/dev/ttyAMA1", 115200) < 0) {
        exit(1);
    }

    /* Write to the serial port */
    if (serial_write(fd, s, sizeof(s)) < 0) {
        exit(1);
    }

    /* Read up to buf size or 2000ms timeout */
    if ((ret = serial_read(fd, buf, sizeof(buf), 2000)) < 0) {
        exit(1);
    }

    printf("read %d bytes: _%s_\n", ret, buf);

    return 0;
}