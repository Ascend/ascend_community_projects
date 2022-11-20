/**
* Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
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

#define LEN_MAX 1024
#define SELF_READ_BUF_MAX 276
#define TIMEOUT_MS 2000
#define SELF_READ_MAX 126

int main(int argc, char const *argv[]) {
    int fd;
    const char *buf = "nihao\r\n";
    uint8_t new[LEN_MAX];
    uint8_t read_buf[SELF_READ_BUF_MAX];
    int i = 0, res = 0;

    memset(new, 0, LEN_MAX);
    memset(read_buf, 0, SELF_READ_BUF_MAX);

    fd = serial_open("/dev/ttyAMA1", BAUDRATES_115200);
    if (fd == -1) { return 0; }

    printf("fd %d\r\n", fd);
    serial_write(fd, buf, strlen(buf));
    res = serial_tostring(fd, new, LEN_MAX);
    printf("%s\n", new);
    res = serial_read(fd, read_buf, SELF_READ_MAX, TIMEOUT_MS);
    printf("%d\r\n", res);
    serial_write(fd, read_buf, sizeof(read_buf));

    memset(read_buf, 0, SELF_READ_BUF_MAX);
    memset(new, 0, LEN_MAX);
    res = serial_tostring(fd, new, SELF_READ_BUF_MAX);
    printf("%s\r\n", new);
    serial_close(&fd);
    return 0;
}
