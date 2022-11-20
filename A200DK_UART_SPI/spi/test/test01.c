#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../src/a200dkspi.h"

int main(void) {
    int fd, res;
    uint8_t new[1024];
    uint8_t buf[4] = {0xaa, 0xbb, 0xcc, 0xdd};

    memset(new, 0, 1024);

    if ((fd = spi_open("/dev/spidev0.0", 0, 1000000)) < 0) {
        printf("failed to open\n");
        exit(1);
    } else {
        printf("%d\n", fd);
    }

    res = spi_tostring(fd, new, 1024);
    printf("%s\n", new);

    if (spi_xfer(fd, buf, buf, sizeof(buf)) < 0) {
        printf("failed to transfer\n");
        exit(1);
    }

    printf("shifted in: 0x%02x 0x%02x 0x%02x 0x%02x\n", buf[0], buf[1], buf[2], buf[3]);

    spi_close(&fd);

    return 0;
}
