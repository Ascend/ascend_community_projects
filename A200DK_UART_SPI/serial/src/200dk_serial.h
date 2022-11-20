#ifndef PERIPHERY_SERIAL_H
#define PERIPHERY_SERIAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAUDRATES_0 0
#define BAUDRATES_50 50
#define BAUDRATES_75 75
#define BAUDRATES_110 110
#define BAUDRATES_134 134
#define BAUDRATES_150 150
#define BAUDRATES_200 200
#define BAUDRATES_300 300
#define BAUDRATES_600 600
#define BAUDRATES_1200 1200
#define BAUDRATES_1800 1800
#define BAUDRATES_2400 2400
#define BAUDRATES_4800 4800
#define BAUDRATES_9600 9600
#define BAUDRATES_19200 19200
#define BAUDRATES_38400 38400
#define BAUDRATES_57600 57600
#define BAUDRATES_115200 115200
#define BAUDRATES_230400 230400
#define BAUDRATES_460800 460800
#define BAUDRATES_500000 500000
#define BAUDRATES_576000 576000
#define BAUDRATES_921600 921600
#define BAUDRATES_1000000 1000000
#define BAUDRATES_1152000 1152000
#define BAUDRATES_1500000 1500000
#define BAUDRATES_2000000 2000000
#define BAUDRATES_2500000 2500000
#define BAUDRATES_3000000 3000000
#define BAUDRATES_3500000 3500000
#define BAUDRATES_4000000 4000000

#define BAUDRATES_BITS_0 B0
#define BAUDRATES_BITS_50 B50
#define BAUDRATES_BITS_75 B75
#define BAUDRATES_BITS_110 B110
#define BAUDRATES_BITS_134 B134
#define BAUDRATES_BITS_150 B150
#define BAUDRATES_BITS_200 B200
#define BAUDRATES_BITS_300 B300
#define BAUDRATES_BITS_600 B600
#define BAUDRATES_BITS_1200 B1200
#define BAUDRATES_BITS_1800 B1800
#define BAUDRATES_BITS_2400 B2400
#define BAUDRATES_BITS_4800 B4800
#define BAUDRATES_BITS_9600 B9600
#define BAUDRATES_BITS_19200 B19200
#define BAUDRATES_BITS_38400 B38400
#define BAUDRATES_BITS_57600 B57600
#define BAUDRATES_BITS_115200 B115200
#define BAUDRATES_BITS_230400 B230400
#define BAUDRATES_BITS_460800 B460800
#define BAUDRATES_BITS_500000 B500000
#define BAUDRATES_BITS_576000 B576000
#define BAUDRATES_BITS_921600 B921600
#define BAUDRATES_BITS_1000000 B1000000
#define BAUDRATES_BITS_1152000 B1152000
#define BAUDRATES_BITS_1500000 B1500000
#define BAUDRATES_BITS_2000000 B2000000
#define BAUDRATES_BITS_2500000 B2500000
#define BAUDRATES_BITS_3000000 B3000000
#define BAUDRATES_BITS_3500000 B3500000
#define BAUDRATES_BITS_4000000 B4000000

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) \
    if (EnableWarnings) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define PARITY_NONE 0
#define PARITY_ODD 1
#define PARITY_EVEN 2

/*	Primary Functions	*/
int serial_open(const char *path, uint32_t baudrate);
int serial_open_advanced(const char *path,
                         uint32_t baudrate, unsigned int databits,
                         uint32_t parity, unsigned int stopbits,
                         bool xonxoff, bool rtscts);
int serial_read(int fd, uint8_t *buf, size_t len, int timeout_ms);
int serial_readline(int fd, uint8_t *buf, size_t maxlen, int timeout_ms);
int serial_write(int fd, uint8_t *buf, size_t len);
int serial_flush(int fd);
int serial_input_waiting(int fd, unsigned int *count);
int serial_output_waiting(int fd, unsigned int *count);
int serial_poll(int fd, int timeout_ms);
int serial_close(int *fd);

/*	Getters	*/
int serial_get_baudrate(int fd, uint32_t *baudrate);
int serial_get_databits(int fd, unsigned int *databits);
int serial_get_parity(int fd, uint32_t *parity);
int serial_get_stopbits(int fd, unsigned int *stopbits);
int serial_get_xonxoff(int fd, bool *xonxoff);
int serial_get_rtscts(int fd, bool *rtscts);
int serial_get_vmin(int fd, unsigned int *vmin);
int serial_get_vtime(int fd, float *vtime);

/*	Setters	*/
int serial_set_baudrate(int fd, uint32_t baudrate);
int serial_set_databits(int fd, unsigned int databits);
int serial_set_parity(int fd, uint32_t parity);
int serial_set_stopbits(int fd, unsigned int stopbits);
int serial_set_xonxoff(int fd, bool enabled);
int serial_set_rtscts(int fd, bool enabled);
int serial_set_vmin(int fd, unsigned int vmin);
int serial_set_vtime(int fd, float vtime);

/*	Miscellaneous	*/
int serial_tostring(int fd, char *str, size_t len);

#ifdef __cplusplus
}
#endif

#endif
