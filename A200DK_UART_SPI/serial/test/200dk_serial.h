#ifndef _PERIPHERY_SERIAL_H
#define _PERIPHERY_SERIAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) if (EnableWarnings) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
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
int serial_get_vtime(int fd, float* vtime);

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

