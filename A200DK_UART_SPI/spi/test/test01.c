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
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "a200dkspi.h"

#define SPI_SPEED 1000000
#define LEN_MAX 1024
#define BUF_LEN 4

int main(void) {
    int fd, res;
    uint8_t new[LEN_MAX];
    uint8_t buf[BUF_LEN] = {0xaa, 0xbb, 0xcc, 0xdd};
    uint8_t rbuf[BUF_LEN] = {0};
    uint8_t i = 0;

    memset(new, 0, LEN_MAX);

    if ((fd = spi_open("/dev/spidev0.0", 0, SPI_SPEED)) < 0) {
        printf("failed to open\n");
        exit(1);
    } else {
        printf("%d\n", fd);
    }

    res = spi_tostring(fd, new, SPI_SPEED);
    printf("%s\n", new);

    if (spi_xfer(fd, buf, rbuf, sizeof(buf)) < 0) {
        printf("failed to transfer\n");
        exit(1);
    }
	
	printf("shifted in: 0x%02x 0x%02x 0x%02x 0x%02x\n", rbuf[0], rbuf[1], rbuf[BUF_LEN - 1], rbuf[BUF_LEN]);

    spi_close(&fd);

    return 0;
}
