#ifndef PERIPHERY_SPI_H
#define PERIPHERY_SPI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) \
    if (EnableWarnings) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define SPIDEV_MAXPATH 4096
#define BLOCK_SIZE_CONTROL_FILE "/sys/module/spidev/parameters/bufsiz"
//	The xfwr3 function attempts to use large blocks if /sys/module/spidev/parameters/bufsiz setting allows it.
//	However where we cannot get a value from that file, we fall back to this safe default.
#define XFER3_DEFAULT_BLOCK_SIZE SPIDEV_MAXPATH
//	Largest block size for xfer3 - even if /sys/module/spidev/parameters/bufsiz allows bigger
//	blocks, we won't go above this value. As I understand, DMA is not used for anything bigger so why bother.
#define XFER3_MAX_BLOCK_SIZE 65535

typedef enum spi_bit_order {
    MSB_FIRST,
    LSB_FIRST,
} spi_bit_order_t;

/*	Primary Functions	*/
int spi_open(const char *path, unsigned int mode, uint32_t max_speed);
int spi_open_advanced(const char *path, unsigned int mode,
                      uint32_t max_speed, spi_bit_order_t bit_order,
                      uint8_t bits_per_word, uint8_t extra_flags);
int spi_open_advanced2(const char *path, unsigned int mode,
                       uint32_t max_speed, spi_bit_order_t bit_order,
                       uint8_t bits_per_word, uint32_t extra_flags);
int spi_xfer(int fd, const uint8_t *txbuf, uint8_t *rxbuf, size_t len);
int spi_xfer2(int fd, const uint8_t *txbuf, uint8_t *rxbuf, size_t len,
              uint16_t delay_usecs, uint8_t bits_per_word, uint32_t speed_hz);
int spi_xfer3(int fd, const uint8_t *txbuf, uint8_t *rxbuf, size_t len,
              uint16_t delay_usecs, uint8_t bits_per_word, uint32_t speed_hz);
int spi_write(int fd, const uint8_t *txbuf, size_t len);
int spi_read(int fd, uint8_t *txbuf, size_t len);
int spi_close(int *fd);

/*	Getters	*/
int spi_get_mode(int fd, uint8_t *mode);
int spi_get_max_speed(int fd, uint32_t *max_speed);
int spi_get_bit_order(int fd, spi_bit_order_t *bit_order);
int spi_get_bits_per_word(int fd, uint8_t *bits_per_word);
int spi_get_extra_flags(int fd, uint8_t *extra_flags);
int spi_get_cshigh(int fd, bool *cs);
int spi_get_loop(int fd, bool *result);
int spi_get_no_cs(int fd, bool *result);

/*	Setters	*/
int spi_set_m(int fd, uint8_t mode);
int spi_set_mode(int fd, unsigned int mode);
int spi_set_max_speed(int fd, uint32_t max_speed);
int spi_set_bit_order(int fd, spi_bit_order_t bit_order);
int spi_set_bits_per_word(int fd, uint8_t bits_per_word);
int spi_set_extra_flags(int fd, uint8_t extra_flags);
int spi_set_cshigh(int fd, bool val);
int spi_set_loop(int fd, bool val);
int spi_set_no_cs(int fd, bool val);

/*	Miscellaneous	*/
uint32_t get_xfer3_block_size(void);
int spi_tostring(int fd, char *str, size_t len);

#ifdef __cplusplus
}
#endif

#endif