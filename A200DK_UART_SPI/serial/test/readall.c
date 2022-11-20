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

int main(int argc, char const *argv[])
{
    int fd;
    const char *buf = "nihao\r\n";
    uint8_t new[1024];
    uint8_t read_buf[256];
    int i=0,res=0;
    memset(new,0,1024);
    memset(read_buf,0,256);
    fd = serial_open("/dev/ttyAMA1",1200);
    if(fd == -1)
        return 0;
    //serial_set_vmin(fd,200);
    //serial_set_vtime(fd,20);
    printf("%d\r\n",fd);
    //write(fd,buf,strlen(buf));
    serial_write(fd, buf, strlen(buf));
    res= serial_tostring(fd,new,1024);
    printf("%s\n",new);
    while(1)
    {
        res = serial_readall(fd, read_buf, sizeof(read_buf));
	if(res>0)
        {
            if(i++>4)
            {
	        break;
	    }
	    printf("%d %d\r\n",i,res);
	    serial_write(fd, read_buf, sizeof(read_buf));
	    memset(read_buf,0,256);
	}
    }
    serial_close(&fd);
    return 0;
}
