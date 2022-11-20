#include <stdio.h>
#include <string.h>
#include "200dk_serial.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>

#include <sys/select.h>
#include <sys/ioctl.h>
#include <termios.h>

int main(int argc, char const *argv[]) {
    int fd;
    const char *buf = "nihao\r\n";
    uint8_t new[1024];
    uint8_t read_buf[276];
    int i = 0, res = 0;

    memset(new, 0, 1024);
    memset(read_buf, 0, 276);

    fd = serial_open("/dev/ttyAMA1", 115200);
    if (fd == -1)
        return 0;
    res = serial_tostring(fd, new, 1024);
    printf("打开的默认配置：%s\n", new);
    printf("fd %d\r\n", fd);

    serial_set_baudrate(fd, 115200);
    serial_set_databits(fd, 7);
    serial_set_parity(fd, PARITY_ODD);
    serial_set_stopbits(fd, 2);
    serial_set_xonxoff(fd, false);
    serial_set_rtscts(fd, false);
    serial_set_vmin(fd, 233);
    serial_set_vtime(fd, 25.5);

    serial_write(fd, buf, sizeof(buf));
    res = serial_readline(fd, read_buf, 276, 2000);
    printf("接收： %d\r\n", res);
    serial_write(fd, read_buf, sizeof(read_buf));

    memset(read_buf, 0, 276);
    memset(new, 0, 1024);
    res = serial_tostring(fd, new, 1024);
    printf("设置后：%s\r\n", new);
    serial_close(&fd);
    return 0;
}
