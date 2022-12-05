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
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <linux/spi/spidev.h>
#include "a200dkspi.h"

static uint32_t xfer3_block_size = 0;
uint32_t get_xfer3_block_size(void) {
    int value;

    if (xfer3_block_size != 0) { return xfer3_block_size; }

    xfer3_block_size = XFER3_DEFAULT_BLOCK_SIZE;

    FILE *file = fopen(BLOCK_SIZE_CONTROL_FILE, "r");
    if (file != NULL) {
        if (fscanf(file, "%d", &value) == 1 && value > 0) {
            if (value <= XFER3_MAX_BLOCK_SIZE) {
                xfer3_block_size = value;
            } else {
                xfer3_block_size = XFER3_MAX_BLOCK_SIZE;
            }
        }
        fclose(file);
    }

    return xfer3_block_size;
}

int spi_open(const char *path, unsigned int mode, uint32_t max_speed) {
    return spi_open_advanced(path, mode, max_speed, MSB_FIRST, 8, 0);
}

int spi_open_advanced(const char *path, unsigned int mode, uint32_t max_speed, spi_bit_order_t bit_order, uint8_t bits_per_word, uint32_t extra_flags) {
    uint32_t data32;
    uint8_t data8;
    int fd;

    if (mode & ~0x3) {
        ERROR_LOG("Invalid mode (can be 0,1,2,3)");
        return -1;
    }
    if (bit_order != MSB_FIRST && bit_order != LSB_FIRST) {
        ERROR_LOG("Invalid bit order (can be MSB_FIRST,LSB_FIRST)");
    }
#ifndef SPI_IOC_WR_MODE32
    if (extra_flags > 0xff) {
        ERROR_LOG("Kernel version does not support 32-bit SPI mode flags");
        return -1;
    }
#endif

    /*	Open device	*/
    if ((fd = open(path, O_RDWR)) < 0) {
        ERROR_LOG("Opening SPI device \"%s\"", path);
        return -1;
    }

    /*	Set mode, bit order, extra flags	*/
#ifndef SPI_IOC_WR_MODE32
    (void)data32;

    data8 = mode | ((bit_order == LSB_FIRST) ? SPI_LSB_FIRST : 0) | extra_flags;
    if (ioctl(fd, SPI_IOC_WR_MODE, &data8) < 0) {
        close(fd);
        ERROR_LOG("Failed to SPI mode");
        return -1;
    }
#else
    if (extra_flags > 0xff) {
        /*	Use 32-bit mode if extra_flags is wider than 8-bits	*/
        data32 = mode | ((bit_order == LSB_FIRST) ? SPI_LSB_FIRST : 0) | extra_flags;
        if (ioctl(fd, SPI_IOC_WR_MODE32, &data32) < 0) {
            close(fd);
            ERROR_LOG("Failed to SPI mode");
            return -1;
        }
    } else {
        /*	Prefer 8-bit mode, in case this library is inadvertently used on an
         * older kernel.	*/
        data8 = mode | ((bit_order == LSB_FIRST) ? SPI_LSB_FIRST : 0) | extra_flags;
        if (ioctl(fd, SPI_IOC_WR_MODE, &data8) < 0) {
            close(fd);
            ERROR_LOG("Failed to SPI mode");
            return -1;
        }
    }
#endif

    /*	Set max speed	*/
    if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &max_speed) < 0) {
        close(fd);
        ERROR_LOG("Failed to set SPI max speed");
        return -1;
    }

    /*	Set bits per word	*/
    if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits_per_word) < 0) {
        close(fd);
        ERROR_LOG("Setting SPI bits per word");
        return -1;
    }

    return fd;
}

int spi_xfer2(int fd, const uint8_t *txbuf, uint8_t *rxbuf, size_t len, uint16_t delay_usecs, uint8_t bits_per_word, uint32_t speed_hz) {
    int status = 0;
    uint32_t max_speed;
    uint8_t bpw, mode;

    if (len <= 0 || len > SPIDEV_MAXPATH) {
        ERROR_LOG("SPI transfer len < 0 or len > SPIDEV_MAXPATH");
        return -1;
    }
    if (spi_get_max_speed(fd, &max_speed) < 0) { return -1; }
    if (spi_get_bits_per_word(fd, &bpw) < 0) { return -1; }

    /*	Prepare SPI transfer structure	*/
#ifdef SPIDEV_SINGLE
    size_t ii;
    struct spi_ioc_transfer *xferptr;
    memset(&xferptr, 0, sizeof(xferptr));

    xferptr = (struct spi_ioc_transfer *)malloc(sizeof(struct spi_ioc_transfer) * len);

    for (ii = 0; ii < len; ii++) {
        xferptr[ii].tx_buf = (unsigned long)&txbuf[ii];
        xferptr[ii].rx_buf = (unsigned long)&rxbuf[ii];
        xferptr[ii].len = 1;
        xferptr[ii].delay_usecs = delay;
        xferptr[ii].speed_hz = speed_hz ? speed_hz : max_speed;
        xferptr[ii].bits_per_word = bits_per_word ? bits_per_word : bpw;
#ifdef SPI_IOC_WR_MODE32
        xferptr[ii].tx_nbits = 0;
#endif
#ifdef SPI_IOC_RD_MODE32
        xferptr[ii].rx_nbits = 0;
#endif
        status = ioctl(fd, SPI_IOC_MESSAGE(len), xferptr);
        free(xferptr);
        if (status < 0) {
            ERROR_LOG("SPI transfer");
            return -1;
        }
    }
#else
    struct spi_ioc_transfer spi_xfer;
    memset(&spi_xfer, 0, sizeof(struct spi_ioc_transfer));

    spi_xfer.tx_buf = (uint64_t)txbuf;
    spi_xfer.rx_buf = (uint64_t)rxbuf;
    spi_xfer.len = len;
    spi_xfer.delay_usecs = delay_usecs;
    spi_xfer.speed_hz = speed_hz ? speed_hz : max_speed;
    spi_xfer.bits_per_word = bits_per_word ? bits_per_word : bpw;
    spi_xfer.cs_change = 0;

    /*	Transfer	*/
    status = ioctl(fd, SPI_IOC_MESSAGE(1), &spi_xfer);
    if (status < 1) {
        ERROR_LOG("SPI transfer");
        return -1;
    }
#endif

    spi_get_mode(fd, &mode);
    if (mode & SPI_CS_HIGH) { status = read(fd, &rxbuf[0], 0); }

    return 0;
}

int spi_xfer(int fd, const uint8_t *txbuf, uint8_t *rxbuf, size_t len) {
    int status = 0;

    if (len <= 0 || len > SPIDEV_MAXPATH) {
        ERROR_LOG("SPI transfer len < 0 or len > SPIDEV_MAXPATH");
        return -1;
    }

    struct spi_ioc_transfer spi_xfer;

    /*	Prepare SPI transfer structure	*/
    memset(&spi_xfer, 0, sizeof(struct spi_ioc_transfer));
    spi_xfer.tx_buf = (uint64_t)txbuf;
    spi_xfer.rx_buf = (uint64_t)rxbuf;
    spi_xfer.len = len;
    spi_xfer.delay_usecs = 0;
    spi_xfer.speed_hz = 0;
    spi_xfer.bits_per_word = 0;
    spi_xfer.cs_change = 0;

    /*	Transfer	*/
    status = ioctl(fd, SPI_IOC_MESSAGE(1), &spi_xfer);
    if (status < 1) {
        ERROR_LOG("SPI transfer");
        return -1;
    }

    uint8_t mode;
    spi_get_mode(fd, &mode);
    if (mode & SPI_CS_HIGH) { status = read(fd, &rxbuf[0], 0); }

    return 0;
}

int spi_xfer3(int fd, const uint8_t *txbuf, uint8_t *rxbuf, size_t len, uint16_t delay_usecs, uint8_t bits_per_word, uint32_t speed_hz) {
    int status = 0;
    struct spi_ioc_transfer spi_xfer;
    size_t bufsize;
    uint32_t max_speed;
    uint8_t bpw, mode;

    if (len <= 0 || len > SPIDEV_MAXPATH) {
        ERROR_LOG("SPI transfer len < 0 or len > SPIDEV_MAXPATH");
        return -1;
    }
    if (spi_get_max_speed(fd, &max_speed) < 0) { return -1; }
    if (spi_get_bits_per_word(fd, &bpw) < 0) { return -1; }

    bufsize = get_xfer3_block_size();
    if (bufsize > len) {
        bufsize = len;
    }

    /*	Prepare SPI transfer structure	*/
    memset(&spi_xfer, 0, sizeof(struct spi_ioc_transfer));
    spi_xfer.tx_buf = (uint64_t)txbuf;
    spi_xfer.rx_buf = (uint64_t)rxbuf;
    spi_xfer.len = bufsize;
    spi_xfer.delay_usecs = delay_usecs;
    spi_xfer.speed_hz = speed_hz ? speed_hz : max_speed;
    spi_xfer.bits_per_word = bits_per_word ? bits_per_word : bpw;
    spi_xfer.cs_change = 0;

    /*	Transfer	*/
    status = ioctl(fd, SPI_IOC_MESSAGE(1), &spi_xfer);
    if (status < 1) {
        ERROR_LOG("SPI transfer");
        return -1;
    }

    spi_get_mode(fd, &mode);
    if (mode & SPI_CS_HIGH) { status = read(fd, &rxbuf[0], 0); }

    return 0;
}

int spi_write(int fd, const uint8_t *txbuf, size_t len) {
    int reback = 0;

    reback = write(fd, txbuf, len);
    if (reback < 0) {
        fprintf(stderr, "spi_write ():errno:%d --%s\n", errno, strerror(errno));
    }

    return reback;
}

int spi_read(int fd, uint8_t *rxbuf, size_t len) {
    int reback = 0;

    reback = read(fd, rxbuf, len);
    if (reback < 0) {
        fprintf(stderr, "spi_read ():errno:%d --%s\n", errno, strerror(errno));
    }
    return reback;
}

int spi_close(int *fd) {
    if (*fd < 0) { return 0; }

    /*	Close fd	*/
    if (close(*fd) < 0) {
        ERROR_LOG("Closing SPI device");
        return -1;
    }
    (*fd) = -1;

    return 0;
}

int spi_set_m(int fd, uint8_t mode) {
    uint8_t test = mode;
    if (ioctl(fd, SPI_IOC_WR_MODE, &test) == -1) {
        ERROR_LOG("failed to set SPI mode");
        return -1;
    }
    if (ioctl(fd, SPI_IOC_RD_MODE, &test) == -1) {
        ERROR_LOG("failed to get SPI mode");
        return -1;
    }
    if (test != mode) {
        return -1;
    }
    return 0;
}

static int spi_get_mode__(int fd, uint8_t *mode) {
    uint8_t data8;

    if (ioctl(fd, SPI_IOC_RD_MODE, &data8) < 0) {
        ERROR_LOG("Getting SPI mode");
        return -1;
    }
    *mode = data8;
    return 0;
}

int spi_get_mode(int fd, uint8_t *mode) {
    uint8_t data8;

    if (spi_get_mode__(fd, &data8)) {
        return -1;
    }
    *mode = data8 & (SPI_CPHA | SPI_CPOL);
    return 0;
}

int spi_get_max_speed(int fd, uint32_t *max_speed) {
    uint32_t data32;

    if (ioctl(fd, SPI_IOC_RD_MAX_SPEED_HZ, &data32) < 0) {
        ERROR_LOG("Getting SPI max speed");
        return -1;
    }

    *max_speed = data32;

    return 0;
}

int spi_get_bit_order(int fd, spi_bit_order_t *bit_order) {
    uint8_t data8;

    if (ioctl(fd, SPI_IOC_RD_LSB_FIRST, &data8) < 0) {
        ERROR_LOG("Getting SPI bit order");
        return -1;
    }

    *bit_order = (data8) ? LSB_FIRST : MSB_FIRST;

    return 0;
}

int spi_get_bits_per_word(int fd, uint8_t *bits_per_word) {
    uint8_t data8;

    if (ioctl(fd, SPI_IOC_RD_BITS_PER_WORD, &data8) < 0) {
        ERROR_LOG("Getting SPI bits per word");
        return -1;
    }

    *bits_per_word = data8;

    return 0;
}

int spi_get_extra_flags(int fd, uint8_t *extra_flags) {
    uint8_t data8;

    if (ioctl(fd, SPI_IOC_RD_MODE, &data8) < 0) {
        ERROR_LOG("Getting SPI mode flags");
        return -1;
    }
    /*	Extra mode flags without mode 0-3 and bit order	*/
    *extra_flags = data8 & ~(SPI_CPOL | SPI_CPHA | SPI_LSB_FIRST);

    return 0;
}

int spi_get_cshigh(int fd, bool *cs) {
    uint8_t mode = 0;
    if (spi_get_mode(fd, &mode) < 0) { return -1; }

    *cs = (mode & SPI_CS_HIGH) ? true : false;

    return 0;
}

int spi_get_loop(int fd, bool *result) {
    uint8_t mode = 0;
    if (spi_get_mode(fd, &mode) < 0) { return -1; }

    *result = (mode & SPI_LOOP) ? true : false;

    return 0;
}

int spi_get_no_cs(int fd, bool *result) {
    uint8_t mode = 0;
    if (spi_get_mode(fd, &mode) < 0) { return -1; }

    *result = (mode & SPI_NO_CS) ? true : false;

    return 0;
}

int spi_set_mode(int fd, unsigned int mode) {
    uint8_t data8;

    if (mode & ~0x3) {
        ERROR_LOG("Invalid mode (can be 0,1,2,3)");
        return -1;
    }

    if (ioctl(fd, SPI_IOC_RD_MODE, &data8) < 0) {
        ERROR_LOG("Getting SPI mode");
        return -1;
    }

    data8 &= ~(SPI_CPOL | SPI_CPHA);
    data8 |= mode;

    if (ioctl(fd, SPI_IOC_WR_MODE, &data8) < 0) {
        ERROR_LOG("Setting SPI mode");
        return -1;
    }

    return 0;
}

int spi_set_bit_order(int fd, spi_bit_order_t bit_order) {
    uint8_t data8;

    if (bit_order != MSB_FIRST && bit_order != LSB_FIRST) {
        ERROR_LOG("Invalid bit order (can be MSB_FIRST,LSB_FIRST)");
        return -1;
    }

    data8 = (bit_order == LSB_FIRST) ? 1 : 0;

    if (ioctl(fd, SPI_IOC_WR_LSB_FIRST, &data8) < 0) {
        ERROR_LOG("Setting SPI bit order");
        return -1;
    }
    return 0;
}

int spi_set_extra_flags(int fd, uint8_t extra_flags) {
    uint8_t data8;

    if (ioctl(fd, SPI_IOC_RD_MODE, &data8) < 0) {
        ERROR_LOG("Getting SPI mode flags");
        return -1;
    }
    /*	Keep mode 0-3 and bit order	*/
    data8 &= (SPI_CPOL | SPI_CPHA | SPI_LSB_FIRST);
    /*	Set extra flags	*/
    data8 |= extra_flags;

    if (ioctl(fd, SPI_IOC_WR_MODE, &data8) < 0) {
        ERROR_LOG("Setting SPI mode flags");
        return -1;
    }
    return 0;
}

int spi_set_max_speed(int fd, uint32_t max_speed) {
    if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &max_speed) < 0) {
        ERROR_LOG("Setting SPI max speed");
        return -1;
    }
    return 0;
}

int spi_set_bits_per_word(int fd, uint8_t bits_per_word) {
    if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits_per_word) < 0) {
        ERROR_LOG("Setting SPI bits per word");
        return -1;
    }
    return 0;
}

int spi_set_cshigh(int fd, bool val) {
    uint8_t tmp, mode = 0;

    if (spi_get_mode(fd, &mode) != 0) { return -1; }

    tmp = (val == true) ? (mode | SPI_CS_HIGH) : (mode & ~SPI_CS_HIGH);

    if (spi_set_m(fd, tmp) < 0) { return -1; }

    return 0;
}

int spi_set_no_cs(int fd, bool val) {
    uint8_t tmp, mode;

    if (spi_get_mode(fd, &mode) < 0) { return -1; }

    tmp = (val == true) ? (mode | SPI_NO_CS) : (mode & ~SPI_NO_CS);

    if (spi_set_m(fd, tmp) < 0) { return -1; }

    return 0;
}

int spi_set_loop(int fd, bool val) {
    uint8_t tmp, mode;

    if (spi_get_mode(fd, &mode) < 0) { return -1; }

    tmp = (val == true) ? (mode | SPI_LOOP) : mode & ~SPI_LOOP;

    if (spi_set_m(fd, tmp) < 0) { return -1; }

    return 0;
}

int spi_tostring(int fd, char *str, size_t len) {
    char mode_str[4];
    uint32_t max_speed;
    char max_speed_str[16];
    uint8_t bits_per_word;
    char bits_per_word_str[4];
    spi_bit_order_t bit_order;
    char bit_order_str[16];
    uint8_t extra_flags, mode;
    char extra_flags_str[4];

    if (spi_get_mode(fd, &mode) < 0) {
        strncpy(mode_str, "?", sizeof(mode_str));
    } else {
        snprintf(mode_str, sizeof(mode_str), "%d", mode);
    }

    if (spi_get_max_speed(fd, &max_speed) < 0) {
        strncpy(max_speed_str, "?", sizeof(max_speed_str));
    } else {
        snprintf(max_speed_str, sizeof(max_speed_str), "%u", max_speed);
    }

    if (spi_get_bit_order(fd, &bit_order) < 0) {
        strncpy(bit_order_str, "?", sizeof(bit_order_str));
    } else {
        strncpy(bit_order_str, (bit_order == LSB_FIRST) ? "LSB first" : "MSB first", sizeof(bit_order_str));
    }

    if (spi_get_bits_per_word(fd, &bits_per_word) < 0) {
        strncpy(bits_per_word_str, "?", sizeof(bits_per_word_str));
    } else {
        snprintf(bits_per_word_str, sizeof(bits_per_word_str), "%u", bits_per_word);
    }

    if (spi_get_extra_flags(fd, &extra_flags) < 0) {
        strncpy(extra_flags_str, "?", sizeof(extra_flags_str));
    } else {
        snprintf(extra_flags_str, sizeof(extra_flags_str), "%02x", extra_flags);
    }

    return snprintf(str, len, "SPI (fd=%d, mode=%s, max_speed=%s, bit_order=%s, bits_per_word=%s, extra_flags=%s)", fd, mode_str, max_speed_str, bit_order_str, bits_per_word_str, extra_flags_str);
}