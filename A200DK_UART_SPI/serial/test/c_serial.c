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
    case 50: return B50;
    case 75: return B75;
    case 110: return B110;
    case 134: return B134;
    case 150: return B150;
    case 200: return B200;
    case 300: return B300;
    case 600: return B600;
    case 1200: return B1200;
    case 1800: return B1800;
    case 2400: return B2400;
    case 4800: return B4800;
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 500000: return B500000;
    case 576000: return B576000;
    case 921600: return B921600;
    case 1000000: return B1000000;
    case 1152000: return B1152000;
    case 1500000: return B1500000;
    case 2000000: return B2000000;
#ifdef B2500000
    case 2500000: return B2500000;
#endif
#ifdef B3000000
    case 3000000: return B3000000;
#endif
#ifdef B3500000
    case 3500000: return B3500000;
#endif
#ifdef B4000000
    case 4000000: return B4000000;
#endif
    default: return -1;
    }
}

static int _serial_bits_to_baudrate(uint32_t bits) {
    switch (bits) {
    case B0: return 0;
    case B50: return 50;
    case B75: return 75;
    case B110: return 110;
    case B134: return 134;
    case B150: return 150;
    case B200: return 200;
    case B300: return 300;
    case B600: return 600;
    case B1200: return 1200;
    case B1800: return 1800;
    case B2400: return 2400;
    case B4800: return 4800;
    case B9600: return 9600;
    case B19200: return 19200;
    case B38400: return 38400;
    case B57600: return 57600;
    case B115200: return 115200;
    case B230400: return 230400;
    case B460800: return 460800;
    case B500000: return 500000;
    case B576000: return 576000;
    case B921600: return 921600;
    case B1000000: return 1000000;
    case B1152000: return 1152000;
    case B1500000: return 1500000;
    case B2000000: return 2000000;
#ifdef B2500000
    case B2500000: return 2500000;
#endif
#ifdef B3000000
    case B3000000: return 3000000;
#endif
#ifdef B3500000
    case B3500000: return 3500000;
#endif
#ifdef B4000000
    case B4000000: return 4000000;
#endif
    default: return -1;
    }
}

int serial_open(const char *path, uint32_t baudrate) {
    return serial_open_advanced(path, baudrate, 8, PARITY_NONE, 1, false, false);
}

int serial_open_advanced(const char *path, uint32_t baudrate, unsigned int databits, uint32_t parity, unsigned int stopbits, bool xonxoff, bool rtscts) {
    struct termios termios_settings;
    int fd;

    /*	Validate args	*/
    if (databits != 5 && databits != 6 && databits != 7 && databits != 8) {
        ERROR_LOG("Invalid data bits (can be 5,6,7,8)");
        return -1;
    }
    if (parity != PARITY_NONE && parity != PARITY_ODD && parity != PARITY_EVEN) {
        ERROR_LOG("Invalid parity (can be PARITY_NONE,PARITY_ODD,PARITY_EVEN)");
        return -1;
    }
    if (stopbits != 1 && stopbits != 2) {
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
    if (parity != PARITY_NONE)
        termios_settings.c_iflag |= INPCK;
    /*	Only use ISTRIP when less than 8 bits as it strips the 8th bit	*/
    if (parity != PARITY_NONE && databits != 8)
        termios_settings.c_iflag |= ISTRIP;
    if (xonxoff)
        termios_settings.c_iflag |= (IXON | IXOFF);

    /*	c_oflag	*/
    termios_settings.c_oflag = 0;

    /*	c_lflag	*/
    termios_settings.c_lflag = 0;

    /*	c_cflag	*/
    /*	Enable receiver, ignore modem control lines	*/
    termios_settings.c_cflag = CREAD | CLOCAL;

    /*	Databits	*/
    if (databits == 5)
        termios_settings.c_cflag |= CS5;
    else if (databits == 6)
        termios_settings.c_cflag |= CS6;
    else if (databits == 7)
        termios_settings.c_cflag |= CS7;
    else if (databits == 8)
        termios_settings.c_cflag |= CS8;

    /*	Parity	*/
    if (parity == PARITY_EVEN)
        termios_settings.c_cflag |= PARENB;
    else if (parity == PARITY_ODD)
        termios_settings.c_cflag |= (PARENB | PARODD);

    /*	Stopbits	*/
    if (stopbits == 2)
        termios_settings.c_cflag |= CSTOPB;

    /*	RTS/CTS	*/
    if (rtscts)
        termios_settings.c_cflag |= CRTSCTS;

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
        if (ret == 0)
            break;

        if ((ret = read(fd, buf + bytes_read, len - bytes_read)) < 0) {
            ERROR_LOG("Reading serial port");
            return -1;
        }

        /*	If we're using VMIN or VMIN+VTIME semantics for end of read, return now	*/
        if (value_use_termios_timeout(fd))
            return ret;

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
        if (ret == 0)
            break;

        if ((ret = read(fd, buf + bytes_read, maxlen - bytes_read)) < 0) {
            ERROR_LOG("Reading serial port");
            return -1;
        }

        /*	If we're using VMIN or VMIN+VTIME semantics for end of read, return now	*/
        if (value_use_termios_timeout(fd))
            return ret;

        /*	Empty read	*/
        if (ret == 0 && maxlen != 0) {
            ERROR_LOG("Reading serial port: unexpected empty read");
            return -1;
        }

        bytes_read += ret;

        if ((buf)[bytes_read] == '\n')
            break;
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

    if (ret)
        return 1;

    /*	Timed out	*/
    return 0;
}

int serial_close(int *fd) {
    if (*fd < 0)
        return 0;

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
    case CS5:
        *databits = 5;
        break;
    case CS6:
        *databits = 6;
        break;
    case CS7:
        *databits = 7;
        break;
    case CS8:
        *databits = 8;
        break;
    }

    return 0;
}

int serial_get_parity(int fd, uint32_t *parity) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes parity");
        return -1;
    }

    if ((termios_settings.c_cflag & PARENB) == 0)
        *parity = PARITY_NONE;
    else if ((termios_settings.c_cflag & PARODD) == 0)
        *parity = PARITY_EVEN;
    else
        *parity = PARITY_ODD;

    return 0;
}

int serial_get_stopbits(int fd, unsigned int *stopbits) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes stopbits");
        return -1;
    }

    if (termios_settings.c_cflag & CSTOPB)
        *stopbits = 2;
    else
        *stopbits = 1;

    return 0;
}

int serial_get_xonxoff(int fd, bool *xonxoff) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes xonxoff");
        return -1;
    }

    if (termios_settings.c_iflag & (IXON | IXOFF))
        *xonxoff = true;
    else
        *xonxoff = false;

    return 0;
}

int serial_get_rtscts(int fd, bool *rtscts) {
    struct termios termios_settings;

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes rtscts");
        return -1;
    }

    if (termios_settings.c_cflag & CRTSCTS)
        *rtscts = true;
    else
        *rtscts = false;

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

    if (databits != 5 && databits != 6 && databits != 7 && databits != 8) {
        ERROR_LOG("Invalid data bits (can be 5,6,7,8)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes databits");
        return -1;
    }

    termios_settings.c_cflag &= ~CSIZE;
    if (databits == 5)
        termios_settings.c_cflag |= CS5;
    else if (databits == 6)
        termios_settings.c_cflag |= CS6;
    else if (databits == 7)
        termios_settings.c_cflag |= CS7;
    else if (databits == 8)
        termios_settings.c_cflag |= CS8;

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
    if (parity != PARITY_NONE)
        termios_settings.c_iflag |= (INPCK | ISTRIP);

    termios_settings.c_cflag &= ~(PARENB | PARODD);
    if (parity == PARITY_EVEN)
        termios_settings.c_cflag |= PARENB;
    else if (parity == PARITY_ODD)
        termios_settings.c_cflag |= (PARENB | PARODD);

    if (tcsetattr(fd, TCSANOW, &termios_settings) < 0) {
        ERROR_LOG("Setting serial port attributes parity");
        return -1;
    }

    return 0;
}

int serial_set_stopbits(int fd, unsigned int stopbits) {
    struct termios termios_settings;

    if (stopbits != 1 && stopbits != 2) {
        ERROR_LOG("Invalid stop bits (can be 1,2)");
        return -1;
    }

    if (tcgetattr(fd, &termios_settings) < 0) {
        ERROR_LOG("Getting serial port attributes stopbits");
        return -1;
    }

    termios_settings.c_cflag &= ~(CSTOPB);
    if (stopbits == 2)
        termios_settings.c_cflag |= CSTOPB;

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
    if (enabled)
        termios_settings.c_iflag |= (IXON | IXOFF);

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
    if (enabled)
        termios_settings.c_cflag |= CRTSCTS;

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

    if (tcgetattr(fd, &termios_settings) < 0)
        return snprintf(str, len, "Serial (baudrate=?, databits=?, parity=?, stopbits=?, xonxoff=?, rtscts=?)");

    baudrate = _serial_bits_to_baudrate(cfgetospeed(&termios_settings));

    switch (termios_settings.c_cflag & CSIZE) {
    case CS5: databits_str = "5"; break;
    case CS6: databits_str = "6"; break;
    case CS7: databits_str = "7"; break;
    case CS8: databits_str = "8"; break;
    default: databits_str = "?";
    }

    if ((termios_settings.c_cflag & PARENB) == 0)
        parity_str = "none";
    else if ((termios_settings.c_cflag & PARODD) == 0)
        parity_str = "even";
    else
        parity_str = "odd";

    if (termios_settings.c_cflag & CSTOPB)
        stopbits_str = "2";
    else
        stopbits_str = "1";

    if (termios_settings.c_iflag & (IXON | IXOFF))
        xonxoff_str = "true";
    else
        xonxoff_str = "false";

    if (termios_settings.c_cflag & CRTSCTS)
        rtscts_str = "true";
    else
        rtscts_str = "false";

    vmin = termios_settings.c_cc[VMIN];
    vtime = ((float)termios_settings.c_cc[VTIME]) / 10;

    return snprintf(str, len, "Serial (fd=%d, baudrate=%u, databits=%s, parity=%s, stopbits=%s, xonxoff=%s, rtscts=%s, vmin=%u, vtime=%.1f)",
                    fd, baudrate, databits_str, parity_str, stopbits_str, xonxoff_str, rtscts_str, vmin, vtime);
}
