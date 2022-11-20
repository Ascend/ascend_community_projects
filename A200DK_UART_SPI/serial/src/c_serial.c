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

#include "200dk_serial.h"

static int _serial_baudrate_to_bits(uint32_t baudrate) {
    switch (baudrate) {
    case BAUDRATES_0: return BAUDRATES_BITS_0;
    case BAUDRATES_50: return BAUDRATES_BITS_50;
    case BAUDRATES_75: return BAUDRATES_BITS_75;
    case BAUDRATES_110: return BAUDRATES_BITS_110;
    case BAUDRATES_134: return BAUDRATES_BITS_134;
    case BAUDRATES_150: return BAUDRATES_BITS_150;
    case BAUDRATES_200: return BAUDRATES_BITS_200;
    case BAUDRATES_300: return BAUDRATES_BITS_300;
    case BAUDRATES_600: return BAUDRATES_BITS_600;
    case BAUDRATES_1200: return BAUDRATES_BITS_1200;
    case BAUDRATES_1800: return BAUDRATES_BITS_1800;
    case BAUDRATES_2400: return BAUDRATES_BITS_2400;
    case BAUDRATES_4800: return BAUDRATES_BITS_4800;
    case BAUDRATES_9600: return BAUDRATES_BITS_9600;
    case BAUDRATES_19200: return BAUDRATES_BITS_19200;
    case BAUDRATES_38400: return BAUDRATES_BITS_38400;
    case BAUDRATES_57600: return BAUDRATES_BITS_57600;
    case BAUDRATES_115200: return BAUDRATES_BITS_115200;
    case BAUDRATES_230400: return BAUDRATES_BITS_230400;
    case BAUDRATES_460800: return BAUDRATES_BITS_460800;
    case BAUDRATES_500000: return BAUDRATES_BITS_500000;
    case BAUDRATES_576000: return BAUDRATES_BITS_576000;
    case BAUDRATES_921600: return BAUDRATES_BITS_921600;
    case BAUDRATES_1000000: return BAUDRATES_BITS_1000000;
    case BAUDRATES_1152000: return BAUDRATES_BITS_1152000;
    case BAUDRATES_1500000: return BAUDRATES_BITS_1500000;
    case BAUDRATES_2000000: return BAUDRATES_BITS_2000000;

#ifdef B2500000
    case BAUDRATES_2500000: return BAUDRATES_BITS_2500000;
#endif
#ifdef B3000000
    case BAUDRATES_3000000: return BAUDRATES_BITS_3000000;
#endif
#ifdef B3500000
    case BAUDRATES_3500000: return BAUDRATES_BITS_3500000;
#endif
#ifdef B4000000
    case BAUDRATES_4000000: return BAUDRATES_BITS_4000000;
#endif
    default: return -1;
    }
}

static int _serial_bits_to_baudrate(uint32_t bits) {
    switch (bits) {
    case BAUDRATES_BITS_0: return BAUDRATES_0;
    case BAUDRATES_BITS_50: return BAUDRATES_50;
    case BAUDRATES_BITS_75: return BAUDRATES_75;
    case BAUDRATES_BITS_110: return BAUDRATES_110;
    case BAUDRATES_BITS_134: return BAUDRATES_134;
    case BAUDRATES_BITS_150: return BAUDRATES_150;
    case BAUDRATES_BITS_200: return BAUDRATES_200;
    case BAUDRATES_BITS_300: return BAUDRATES_300;
    case BAUDRATES_BITS_600: return BAUDRATES_600;
    case BAUDRATES_BITS_1200: return BAUDRATES_1200;
    case BAUDRATES_BITS_1800: return BAUDRATES_1800;
    case BAUDRATES_BITS_2400: return BAUDRATES_2400;
    case BAUDRATES_BITS_4800: return BAUDRATES_4800;
    case BAUDRATES_BITS_9600: return BAUDRATES_9600;
    case BAUDRATES_BITS_19200: return BAUDRATES_19200;
    case BAUDRATES_BITS_38400: return BAUDRATES_38400;
    case BAUDRATES_BITS_57600: return BAUDRATES_57600;
    case BAUDRATES_BITS_115200: return BAUDRATES_115200;
    case BAUDRATES_BITS_230400: return BAUDRATES_230400;
    case BAUDRATES_BITS_460800: return BAUDRATES_460800;
    case BAUDRATES_BITS_500000: return BAUDRATES_500000;
    case BAUDRATES_BITS_576000: return BAUDRATES_576000;
    case BAUDRATES_BITS_921600: return BAUDRATES_921600;
    case BAUDRATES_BITS_1000000: return BAUDRATES_1000000;
    case BAUDRATES_BITS_1152000: return BAUDRATES_1152000;
    case BAUDRATES_BITS_1500000: return BAUDRATES_1500000;
    case BAUDRATES_BITS_2000000: return BAUDRATES_2000000;

#ifdef B2500000
    case BAUDRATES_BITS_2500000: return BAUDRATES_2500000;
#endif
#ifdef B3000000
    case BAUDRATES_BITS_3000000: return BAUDRATES_3000000;
#endif
#ifdef B3500000
    case BAUDRATES_BITS_3500000: return BAUDRATES_3500000;
#endif
#ifdef B4000000
    case BAUDRATES_BITS_4000000: return BAUDRATES_4000000;
#endif
    default: return -1;
    }
}

int serial_open(const char *path, uint32_t baudrate) {
    return serial_open_advanced(path, baudrate, DATABITS_8, PARITY_NONE, STOPBITS_1, false, false);
}

int serial_open_advanced(const char *path, uint32_t baudrate, unsigned int databits, uint32_t parity, unsigned int stopbits, bool xonxoff, bool rtscts) {
    struct termios termios_settings;
    int fd;

    /*	Validate args	*/
    if (databits != DATABITS_5 && databits != DATABITS_6 && databits != DATABITS_7 && databits != DATABITS_8) {
        ERROR_LOG("Invalid data bits (can be 5,6,7,8)");
        return -1;
    }
    if (parity != PARITY_NONE && parity != PARITY_ODD && parity != PARITY_EVEN) {
        ERROR_LOG("Invalid parity (can be PARITY_NONE,PARITY_ODD,PARITY_EVEN)");
        return -1;
    }
    if (stopbits != STOPBITS_1 && stopbits != STOPBITS_2) {
        ERROR_LOG("Invalid stop bits (can be 1,2)");
        return -1;
    }

    /*	Open serial port	*/
    if ((fd = open(path, O_RDWR | O_NOCTTY)) < 0) {
        ERROR_LOG("Opening serial port \"%s\"", path);
        return -1;
    }

    memset(&termios_settings, 0, sizeof(termios_settings));

    /*	c_iflag	*/

    /*	Ignore break characters	*/
    termios_settings.c_iflag = IGNBRK;
    if (parity != PARITY_NONE) {
        termios_settings.c_iflag |= INPCK;
    }
    /*	Only use ISTRIP when less than 8 bits as it strips the 8th bit	*/
    if (parity != PARITY_NONE && databits != DATABITS_8) {
        termios_settings.c_iflag |= ISTRIP;
    }
    if (xonxoff) { termios_settings.c_iflag |= (IXON | IXOFF); }

    /*	c_oflag	*/
    termios_settings.c_oflag = 0;

    /*	c_lflag	*/
    termios_settings.c_lflag = 0;

    /*	c_cflag	*/
    /*	Enable receiver, ignore modem control lines	*/
    termios_settings.c_cflag = CREAD | CLOCAL;

    /*	Databits	*/
    if (databits == DATABITS_5) {
        termios_settings.c_cflag |= CS5;
    } else if (databits == DATABITS_6) {
        termios_settings.c_cflag |= CS6;
    } else if (databits == DATABITS_7) {
        termios_settings.c_cflag |= CS7;
    } else if (databits == DATABITS_8) {
        termios_settings.c_cflag |= CS8;
    }

    /*	Parity	*/
    if (parity == PARITY_EVEN) {
        termios_settings.c_cflag |= PARENB;
    } else if (parity == PARITY_ODD) {
        termios_settings.c_cflag |= (PARENB | PARODD);
    }

    /*	Stopbits	*/
    if (stopbits == STOPBITS_2) { termios_settings.c_cflag |= CSTOPB; }

    /*	RTS/CTS	*/
    if (rtscts) { termios_settings.c_cflag |= CRTSCTS; }

    /*	Baudrate	*/
    cfsetispeed(&termios_settings, _serial_baudrate_to_bits(baudrate));
    cfsetospeed(&termios_settings, _serial_baudrate_to_bits(baudrate));

    /*	Set termios attributes	*/
    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        close(fd);
        fd = -1;
        ERROR_LOG("Setting serial port attributes");
        return -1;
    }
    return fd;
}

int serial_get_vmin(int fd, unsigned int *vmin) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes vmin");
        return -1;
    }

    *vmin = termios_settings.c_cc[VMIN];

    return 0;
}

bool value_use_termios_timeout(int fd) {
    unsigned int vmin = 0;
    if (serial_get_vmin(fd, &vmin) < 0) {
        ERROR_LOG("Failed to get value_use_termios_timeout");
        return false;
    }
    return vmin > 0;
}

int serial_read(int fd, uint8_t *buf, size_t len, int timeout_ms) {
    ssize_t ret;

    struct timeval tv_timeout;
    tv_timeout.tv_sec = timeout_ms / 1000;
    tv_timeout.tv_usec = (timeout_ms % 1000) * 1000;

    size_t bytes_read = 0;

    while (bytes_read < len) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        if ((ret = select(fd + 1, &rfds, NULL, NULL, (timeout_ms < 0) ? NULL : &tv_timeout)) < 0) {
            ERROR_LOG("select() on serial port");
            return -1;
        }
        /*	Timeout	*/
        if (ret == 0) { break; }

        if ((ret = read(fd, buf + bytes_read, len - bytes_read)) < 0) {
            ERROR_LOG("Reading serial port");
            return -1;
        }

        /*	If we're using VMIN or VMIN+VTIME semantics for end of read, return now	*/
        if (value_use_termios_timeout(fd)) { return ret; }

        /*	Empty read	*/
        if (ret == 0 && len != 0) {
            ERROR_LOG("Reading serial port: unexpected empty read");
            return -1;
        }

        bytes_read += ret;
    }

    return bytes_read;
}

int serial_readline(int fd, uint8_t *buf, size_t maxlen, int timeout_ms) {
    ssize_t ret;

    struct timeval tv_timeout;
    tv_timeout.tv_sec = timeout_ms / 1000;
    tv_timeout.tv_usec = (timeout_ms % 1000) * 1000;

    size_t bytes_read = 0;

    while (bytes_read < maxlen) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        if ((ret = select(fd + 1, &rfds, NULL, NULL, (timeout_ms < 0) ? NULL : &tv_timeout)) < 0) {
            ERROR_LOG("select() on serial port");
            return -1;
        }
        /*	Timeout	*/
        if (ret == 0) { break; }

        if ((ret = read(fd, buf + bytes_read, maxlen - bytes_read)) < 0) {
            ERROR_LOG("Reading serial port");
            return -1;
        }

        /*	If we're using VMIN or VMIN+VTIME semantics for end of read, return now	*/
        if (value_use_termios_timeout(fd)) { return ret; }

        /*	Empty read	*/
        if (ret == 0 && maxlen != 0) {
            ERROR_LOG("Reading serial port: unexpected empty read");
            return -1;
        }

        bytes_read += ret;

        if ((buf)[bytes_read] == '\n') { break; }
    }

    return bytes_read;
}

int serial_write(int fd, uint8_t *buf, size_t len) {
    ssize_t ret;

    if ((ret = write(fd, buf, len)) < 0) {
        ERROR_LOG("Writing serial port");
        return -1;
    }

    return ret;
}

int serial_flush(int fd) {
    if (tcdrain(fd) < 0) {
        ERROR_LOG("Flushing serial port");
        return -1;
    }

    return 0;
}

int serial_input_waiting(int fd, unsigned int *count) {
    if (ioctl(fd, TIOCINQ, count) < 0) {
        ERROR_LOG("TIOCINQ query");
        return -1;
    }

    return 0;
}

int serial_output_waiting(int fd, unsigned int *count) {
    if (ioctl(fd, TIOCOUTQ, count) < 0) {
        ERROR_LOG("TIOCOUTQ query");
        return -1;
    }

    return 0;
}

int serial_poll(int fd, int timeout_ms) {
    struct pollfd fds[1];
    int ret;

    /*	Poll	*/
    fds[0].fd = fd;
    fds[0].events = POLLIN | POLLPRI;
    if ((ret = poll(fds, 1, timeout_ms)) < 0) {
        ERROR_LOG("Polling serial port timeout_ms%d", timeout_ms);
        return -1;
    }

    if (ret) {
        return 1;
    }

    /*	Timed out	*/
    return 0;
}

int serial_close(int *fd) {
    if (*fd < 0) {
        return 0;
    }

    if (close(*fd) < 0) {
        ERROR_LOG("Closing serial port");
        return -1;
    }

    *fd = -1;

    return 0;
}

int serial_get_baudrate(int fd, uint32_t *baudrate) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes baudrate");
        return -1;
    }

    *baudrate = _serial_bits_to_baudrate(cfgetospeed(&termios_settings));

    return 0;
}

int serial_get_databits(int fd, unsigned int *databits) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes databits");
        return -1;
    }

    switch (termios_settings.c_cflag & CSIZE) {
    case CS5: *databits = DATABITS_5; break;
    case CS6: *databits = DATABITS_6; break;
    case CS7: *databits = DATABITS_7; break;
    case CS8: *databits = DATABITS_8; break;
    default: break;
    }

    return 0;
}

int serial_get_parity(int fd, uint32_t *parity) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes parity");
        return -1;
    }

    if ((termios_settings.c_cflag & PARENB) == 0) {
        *parity = PARITY_NONE;
    } else if ((termios_settings.c_cflag & PARODD) == 0) {
        *parity = PARITY_EVEN;
    } else {
        *parity = PARITY_ODD;
    }

    return 0;
}

int serial_get_stopbits(int fd, unsigned int *stopbits) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes stopbits");
        return -1;
    }

    *stopbits = (termios_settings.c_cflag & CSTOPB) ? STOPBITS_2 : STOPBITS_1;

    return 0;
}

int serial_get_xonxoff(int fd, bool *xonxoff) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes xonxoff");
        return -1;
    }

    *xonxoff = (termios_settings.c_iflag & (IXON | IXOFF)) ? true : false;

    return 0;
}

int serial_get_rtscts(int fd, bool *rtscts) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes rtscts");
        return -1;
    }

    *rtscts = (termios_settings.c_cflag & CRTSCTS) ? true : false;

    return 0;
}

int serial_get_vtime(int fd, float *vtime) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes vtime");
        return -1;
    }

    *vtime = ((float)termios_settings.c_cc[VTIME]) / 10;

    return 0;
}

int serial_set_baudrate(int fd, uint32_t baudrate) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes baudrate");
        return -1;
    }

    cfsetispeed(&termios_settings, _serial_baudrate_to_bits(baudrate));
    cfsetospeed(&termios_settings, _serial_baudrate_to_bits(baudrate));

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes baudrate");
        return -1;
    }

    return 0;
}

int serial_set_databits(int fd, unsigned int databits) {
    struct termios termios_settings;

    if (databits != DATABITS_5 && databits != DATABITS_6 && databits != DATABITS_7 && databits != DATABITS_8) {
        ERROR_LOG("Invalid data bits (can be 5,6,7,8)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes databits");
        return -1;
    }

    termios_settings.c_cflag &= ~CSIZE;
    switch (databits) {
    case DATABITS_5: termios_settings.c_cflag |= CS5; break;
    case DATABITS_6: termios_settings.c_cflag |= CS6; break;
    case DATABITS_7: termios_settings.c_cflag |= CS7; break;
    case DATABITS_8: termios_settings.c_cflag |= CS8; break;
    default:
        break;
    }

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes databits");
        return -1;
    }

    return 0;
}

int serial_set_parity(int fd, uint32_t parity) {
    struct termios termios_settings;

    if (parity != PARITY_NONE && parity != PARITY_ODD && parity != PARITY_EVEN) {
        ERROR_LOG("Invalid parity (can be PARITY_NONE,PARITY_ODD,PARITY_EVEN)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes parity");
        return -1;
    }

    termios_settings.c_iflag &= ~(INPCK | ISTRIP);
    if (parity != PARITY_NONE) { termios_settings.c_iflag |= (INPCK | ISTRIP); }

    termios_settings.c_cflag &= ~(PARENB | PARODD);
    if (parity == PARITY_EVEN) {
        termios_settings.c_cflag |= PARENB;
    } else if (parity == PARITY_ODD) {
        termios_settings.c_cflag |= (PARENB | PARODD);
    }

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Setting serial port attributes parity");
        return -1;
    }

    return 0;
}

int serial_set_stopbits(int fd, unsigned int stopbits) {
    struct termios termios_settings;

    if (stopbits != 1 && stopbits != STOPBITS_2) {
        ERROR_LOG("Invalid stop bits (can be 1,2)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes stopbits");
        return -1;
    }

    termios_settings.c_cflag &= ~(CSTOPB);
    if (stopbits == STOPBITS_2) { termios_settings.c_cflag |= CSTOPB; }

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Setting serial port attributes stopbits");
        return -1;
    }

    return 0;
}

int serial_set_xonxoff(int fd, bool enabled) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes enabled");
        return -1;
    }

    termios_settings.c_iflag &= ~(IXON | IXOFF | IXANY);
    if (enabled) { termios_settings.c_iflag |= (IXON | IXOFF); }

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes enabled");
        return -1;
    }

    return 0;
}

int serial_set_rtscts(int fd, bool enabled) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes enabled");
        return -1;
    }

    termios_settings.c_cflag &= ~CRTSCTS;
    if (enabled) { termios_settings.c_cflag |= CRTSCTS; }

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes enabled");
        return -1;
    }

    return 0;
}

int serial_set_vmin(int fd, unsigned int vmin) {
    struct termios termios_settings;

    if (vmin > 255) {
        ERROR_LOG("Invalid vmin (can be 0-255)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes vmin");
        return -1;
    }

    termios_settings.c_cc[VMIN] = vmin;

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes vmin");
        return -1;
    }
    return 0;
}

int serial_set_vtime(int fd, float vtime) {
    struct termios termios_settings;

    if (vtime < 0.0 || vtime > 25.5) {
        ERROR_LOG("Invalid vtime (can be 0-25.5)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes vtime");
        return -1;
    }

    termios_settings.c_cc[VTIME] = ((unsigned int)(vtime * 10));

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes vtime");
        return -1;
    }

    return 0;
}

int serial_tostring(int fd, char *str, size_t len) {
    struct termios termios_settings;
    uint32_t baudrate;
    const char *databits_str, *parity_str, *stopbits_str, *xonxoff_str, *rtscts_str;
    unsigned int vmin;
    float vtime;

    /*	Instead of calling all of our individual getter functions, let's poll
     * termios attributes once to be efficient.	*/

    if (tcgetattr(fd, &termios_settings) < 0) {
        return snprintf(str, len, "Serial (baudrate=?, databits=?, parity=?, stopbits=?, xonxoff=?, rtscts=?)");
    }

    baudrate = _serial_bits_to_baudrate(cfgetospeed(&termios_settings));

    switch (termios_settings.c_cflag & CSIZE) {
    case CS5: databits_str = "5"; break;
    case CS6: databits_str = "6"; break;
    case CS7: databits_str = "7"; break;
    case CS8: databits_str = "8"; break;
    default: databits_str = "?";
    }

    if ((termios_settings.c_cflag & PARENB) == 0) {
        parity_str = "none";
    } else if ((termios_settings.c_cflag & PARODD) == 0) {
        parity_str = "even";
    } else {
        parity_str = "odd";
    }

    stopbits_str = (termios_settings.c_cflag & CSTOPB) ? "2" : "1";
    xonxoff_str = (termios_settings.c_iflag & (IXON | IXOFF)) ? "true" : "false";
    rtscts_str = (termios_settings.c_cflag & CRTSCTS) ? "true" : "false";
    vmin = termios_settings.c_cc[VMIN];
    vtime = ((float)termios_settings.c_cc[VTIME]) / 10;

    return snprintf(str, len, "Serial (fd=%d, baudrate=%u, databits=%s, parity=%s, stopbits=%s, xonxoff=%s, rtscts=%s, vmin=%u, vtime=%.1f)",
                    fd, baudrate, databits_str, parity_str, stopbits_str, xonxoff_str, rtscts_str, vmin, vtime);
}
