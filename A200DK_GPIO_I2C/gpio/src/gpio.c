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
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include "gpio.h"

/*
 * i2c_write, for configure PCA6416 register.
 */
int i2c_write(unsigned char slave, unsigned char reg, unsigned char value)
{
    int ret;
    struct i2c_rdwr_ioctl_data ssm_msg = {0};
    unsigned char buf[2] = {0};
    ssm_msg.nmsgs = 1;
    ssm_msg.msgs = (struct i2c_msg *)malloc(ssm_msg.nmsgs * sizeof(struct i2c_msg));
    if (ssm_msg.msgs == NULL) {
        ERROR_LOG("Memory alloc error!");
        return -1;
    }
    buf[0] = reg;
    buf[1] = value;
    (ssm_msg.msgs[0]).flags = 0;
    (ssm_msg.msgs[0]).addr = (unsigned short)slave;
    (ssm_msg.msgs[0]).buf = buf;
    (ssm_msg.msgs[0]).len = I2C_WRITE_MSG_LEN;
    ret = ioctl(fd_gpio, I2C_RDWR, &ssm_msg);
    if (ret < 0) {
        ERROR_LOG("write error, ret=%#x, errorno=%#x, %s!", ret, errno, strerror(errno));
        free(ssm_msg.msgs);
        ssm_msg.msgs = NULL;
        return -1;
    }
    free(ssm_msg.msgs);
    ssm_msg.msgs = NULL;
    return 0;
}

/*
 * i2c_read, for reading PCA6416 register.
 */
int i2c_read(unsigned char slave, unsigned char reg, unsigned char *buf)
{
    int ret;
    struct i2c_rdwr_ioctl_data ssm_msg = {0};
    unsigned char regs[2] = {0};
    regs[0] = reg;
    regs[1] = reg;
    ssm_msg.nmsgs = I2C_READ_MSG_NUMS;
    ssm_msg.msgs = (struct i2c_msg *)malloc(ssm_msg.nmsgs * sizeof(struct i2c_msg));
    if (ssm_msg.msgs == NULL) {
        ERROR_LOG("Memory alloc error!");
        return -1;
    }
    (ssm_msg.msgs[0]).flags = 0;
    (ssm_msg.msgs[0]).addr = slave;
    (ssm_msg.msgs[0]).buf = regs;
    (ssm_msg.msgs[0]).len = 1;
    (ssm_msg.msgs[1]).flags = I2C_M_RD;
    (ssm_msg.msgs[1]).addr = slave;
    (ssm_msg.msgs[1]).buf = buf;
    (ssm_msg.msgs[1]).len = 1;
    ret = ioctl(fd_gpio, I2C_RDWR, &ssm_msg);
    if (ret < 0) {
        ERROR_LOG("read data error,ret=%#x !", ret);
        free(ssm_msg.msgs);
        ssm_msg.msgs = NULL;
        return -1;
    }
    free(ssm_msg.msgs);
    ssm_msg.msgs = NULL;
    return 0;
}

/* access i2c_1 device */
int gpio_init(void)
{
    // open i2c-1 device
    fd_gpio = open(I2C1_DEV_NAME, O_RDWR);
    if (fd_gpio < 0) {
        ERROR_LOG("can't open i2c-1!");
        return -1;
    }
    // set i2c-1 retries time
    if (ioctl(fd_gpio, I2C_RETRIES, 1) < 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("set i2c retry fail!");
        return -1;
    }
    // set i2c-1 timeout time, 10ms as unit
    if (ioctl(fd_gpio, I2C_TIMEOUT, 1) < 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("set i2c timeout fail!");
        return -1;
    }
    cleanup_all();
    setwarnings(1);
    return 0;
}

/* close i2c_1 device */
int gpio_close(void)
{
  cleanup_all();
  int error;
  error = close(fd_gpio);
  if (error < 0) {
        ERROR_LOG("can't close i2c-1!");
        return -1;
  }
  fd_gpio = 0;
  return 0;
}

/* GPIO 3,4,5,6,7 */
int pca6416_setup(int gpio, int direction)
{
    unsigned char slave;
    unsigned char reg;
    unsigned char data;
    int ret;
    
    slave = PCA6416_SLAVE_ADDR;
    reg   = PCA6416_GPIO_CFG_REG;
    data  = 0;
    ret = i2c_read(slave, reg, &data);
    if (ret != 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("read i2c-1 %#x %#x to %#x fail!", slave, data, reg);
        return -1;
    }
    switch (direction)
    {
        case INPUT:
            switch (gpio)
            {
                case GPIO3:
                    data |= GPIO3_MASK;
                    break;
                case GPIO4:
                    data |= GPIO4_MASK;
                    break;
                case GPIO5:
                    data |= GPIO5_MASK;
                    break;
                case GPIO6:
                    data |= GPIO6_MASK;
                    break;
                case GPIO7:
                    data |= GPIO7_MASK;
                    break;
                default:
                    ERROR_LOG("wrong gpio number of PCA6416!");
                    return -1;
            }
            break;
        case OUTPUT:
            switch (gpio)
            {
                case GPIO3:
                    data &= ~GPIO3_MASK;
                    break;
                case GPIO4:
                    data &= ~GPIO4_MASK;
                    break;
                case GPIO5:
                    data &= ~GPIO5_MASK;
                    break;
                case GPIO6:
                    data &= ~GPIO6_MASK;
                    break;
                case GPIO7:
                    data &= ~GPIO7_MASK;
                    break;
                default:
                    ERROR_LOG("wrong gpio number of PCA6416!");
                    return -1;
            }
            break;
        default:
            ERROR_LOG("wrong direction value!");
            return -1;
    }
    ret = i2c_write(slave, reg, data);
    if (ret != 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("write i2c-1 %#x %#x to %#x fail!", slave, data, reg);
        return -1;
    }
    return 0;
}

/* GPIO 0,1 */
int ascend310_setup(int gpio, int direction)
{
    int fd_direction;

    switch (gpio)
    {
        case GPIO0:
            fd_direction = open(ASCEND310_GPIO_0_DIR, O_WRONLY);
            break;
        case GPIO1:
            fd_direction = open(ASCEND310_GPIO_1_DIR, O_WRONLY);
            break;
        default:
            ERROR_LOG("wrong gpio number of ascend310!");
            return -1;
    }

    if (-1==fd_direction)
    {
        ERROR_LOG("open gpio DIR file error gpio=%d", gpio);
        return -1;
    }

    switch (direction)
    {
        case INPUT:
            if (-1==write(fd_direction, "in", sizeof("in")))
            {
                ERROR_LOG("gpio write operation error gpio=%d", gpio);
                close(fd_direction);
                return -1;
            }
            break;
        case OUTPUT:
            if (-1==write(fd_direction, "out", sizeof("out")))
            {
                ERROR_LOG("gpio write operation error gpio=%d", gpio);
                close(fd_direction);
                return -1;
            }
            break;
        default:
            ERROR_LOG("wrong direction value!");
            close(fd_direction);
            return -1;
    }
    close(fd_direction);
    return 0;
}

int setup(int gpio, int direction)
{
    if (gpio==GPIO0 || gpio==GPIO1)
    {
        if (ifsetup[gpio])
        {
            WARN_LOG("GPIO%d is already in use, continuing anyway. Use setwarnings(0) to disable warnings.", gpio);
        }
        else
        {
            ifsetup[gpio] = 1;
        }
        return ascend310_setup(gpio, direction);
    }
    else if (gpio>=GPIO3 && gpio<=GPIO7)
    {
        if (ifsetup[gpio])
        {
            WARN_LOG("GPIO%d is already in use, continuing anyway. Use setwarnings(0) to disable warnings.", gpio);
        }
        else
        {
            ifsetup[gpio] = 1;
        }
        return pca6416_setup(gpio, direction);
    }
    else
    {
        ERROR_LOG("invalid gpio number!");
        return -1;
    }
}

int pca6416_output(int gpio, int value)
{
    if (pca6416_gpio_function(gpio)==0)
    {
        ERROR_LOG("can't use ouput while direction=INPUT!");
        return -1;
    }
    
    unsigned char slave;
    unsigned char reg;
    unsigned char data;
    int ret;
    
    slave = PCA6416_SLAVE_ADDR;
    reg   = PCA6416_GPIO_OUT_REG;
    data  = 0;
    ret = i2c_read(slave, reg, &data);
    if (ret != 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("read i2c-1 %#x %#x to %#x fail!", slave, data, reg);
        return -1;
    }
    switch (value)
    {
        case LOW:
            switch (gpio)
            {
                case GPIO3:
                    data &= ~GPIO3_MASK;
                    break;
                case GPIO4:
                    data &= ~GPIO4_MASK;
                    break;
                case GPIO5:
                    data &= ~GPIO5_MASK;
                    break;
                case GPIO6:
                    data &= ~GPIO6_MASK;
                    break;
                case GPIO7:
                    data &= ~GPIO7_MASK;
                    break;
                default:
                    ERROR_LOG("wrong gpio number of PCA6416!");
                    return -1;
            }
            break;
        case HIGH:
            switch (gpio)
            {
                case GPIO3:
                    data |= GPIO3_MASK;
                    break;
                case GPIO4:
                    data |= GPIO4_MASK;
                    break;
                case GPIO5:
                    data |= GPIO5_MASK;
                    break;
                case GPIO6:
                    data |= GPIO6_MASK;
                    break;
                case GPIO7:
                    data |= GPIO7_MASK;
                    break;
                default:
                    ERROR_LOG("wrong gpio number of PCA6416!");
                    return -1;
            }
            break;
        default:
            ERROR_LOG("wrong value number!");
            return -1;
    }
    ret = i2c_write(slave, reg, data);
    if (ret != 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("write i2c-1 %#x %#x to %#x fail!\n", slave, data, reg);
        return -1;
    }
    return 0;
}

int ascend310_output(int gpio, int value)
{
    if (ascend310_gpio_function(gpio)==0)
    {
        ERROR_LOG("Can't use ouput while direction=INPUT!");
        return -1;
    }
    
    int fd_output;

    switch (gpio) {
        case GPIO0:
            fd_output = open(ASCEND310_GPIO_0_VAL, O_WRONLY);
            break;
        case GPIO1:
            fd_output = open(ASCEND310_GPIO_1_VAL, O_WRONLY);
            break;
        default:
            ERROR_LOG("wrong gpio number of ascend310!");
            return -1;
    }

    if (-1==fd_output)
    {
        ERROR_LOG("open gpio DIR file error gpio=%d", gpio);
        return -1;
    }

    switch (value)
    {
        case LOW:
            if (-1==write(fd_output, "0", sizeof("0")))
            {
                ERROR_LOG("gpio write operation error gpio=%d", gpio);
                close(fd_output);
                return -1;
            }
            break;
        case HIGH:
            if (-1==write(fd_output, "1", sizeof("1")))
            {
                ERROR_LOG("gpio write operation error gpio=%d", gpio);
                close(fd_output);
                return -1;
            }
            break;
        default:
            ERROR_LOG("wrong value number!");
            close(fd_output);
            return -1;
    }
    close(fd_output);
    return 0;
}

int output(int gpio, int value)
{
    if (gpio==GPIO0 || gpio==GPIO1)
    {
        return ascend310_output(gpio, value);
    }
    else if (gpio>=GPIO3 && gpio<=GPIO7)
    {
        return pca6416_output(gpio, value);
    }
    else
    {
        ERROR_LOG("invalid gpio number!");
        return -1;
    }
}

int pca6416_input(int gpio)
{
    unsigned char slave;
    unsigned char reg;
    unsigned char data;
    int ret;
    
    slave = PCA6416_SLAVE_ADDR;
    reg   = PCA6416_GPIO_IN_REG;
    data  = 0;
    ret = i2c_read(slave, reg, &data);
    if (ret != 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("read i2c-1 %#x %#x to %#x fail!", slave, data, reg);
        return -1;
    }
    switch (gpio)
    {
        case GPIO3:
            data &= GPIO3_MASK;
            break;
        case GPIO4:
            data &= GPIO4_MASK;
            break;
        case GPIO5:
            data &= GPIO5_MASK;
            break;
        case GPIO6:
            data &= GPIO6_MASK;
            break;
        case GPIO7:
            data &= GPIO7_MASK;
            break;
        default:
            ERROR_LOG("wrong gpio number of PCA6416!");
            return -1;
    }
    if (data > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int ascend310_input(int gpio)
{
    int fd_input;
    char value_str[3];
    int value;
    int len = 3;
    
    switch (gpio) {
        case GPIO0:
            fd_input = open(ASCEND310_GPIO_0_VAL, O_RDONLY);
            break;
        case GPIO1:
            fd_input = open(ASCEND310_GPIO_1_VAL, O_RDONLY);
            break;
        default:
            ERROR_LOG("wrong gpio number of ascend310!");
            return -1;
    }

    if (-1==fd_input)
    {
        ERROR_LOG("open gpio DIR file error gpio=%d", gpio);
        return -1;
    }

    if (-1==read(fd_input, value_str, len))
    {
        ERROR_LOG("gpio read operation error gpio=%d", gpio);
        close(fd_input);
        return -1;
    }
    close(fd_input);
    value = atoi(value_str);
    if (value==HIGH)
    {
        return 1;
    }
    else if (value==LOW)
    {
        return 0;
    }
    else
    {
        ERROR_LOG("invalid value!");
        return -1;
    }
}

int input(int gpio)
{
    if (gpio==GPIO0 || gpio==GPIO1)
    {
        return ascend310_input(gpio);
    }
    else if (gpio>=GPIO3 && gpio<=GPIO7)
    {
        return pca6416_input(gpio);
    }
    else
    {
        ERROR_LOG("invalid gpio number!");
        return -1;
    }
}

/* 0 for input, 1 for output */
int pca6416_gpio_function(int gpio)
{
    unsigned char slave;
    unsigned char reg;
    unsigned char data;
    int ret;
    
    slave = PCA6416_SLAVE_ADDR;
    reg   = PCA6416_GPIO_CFG_REG;
    data  = 0;
    ret = i2c_read(slave, reg, &data);
    if (ret != 0) {
        close(fd_gpio);
        fd_gpio = 0;
        ERROR_LOG("read i2c-1 %#x %#x to %#x fail!", slave, data, reg);
        return -1;
    }
    switch (gpio)
    {
        case GPIO3:
            data &= GPIO3_MASK;
            break;
        case GPIO4:
            data &= GPIO4_MASK;
            break;
        case GPIO5:
            data &= GPIO5_MASK;
            break;
        case GPIO6:
            data &= GPIO6_MASK;
            break;
        case GPIO7:
            data &= GPIO7_MASK;
            break;
        default:
            ERROR_LOG("wrong gpio number of PCA6416!");
            return -1;
    }
    if (data > 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

int ascend310_gpio_function(int gpio)
{
    int fd_function;
    char direction[3];
    int len = 3;
    int index_0 = 0, index_1 = 1, index_2 = 2;
    
    switch (gpio) {
        case GPIO0:
            fd_function = open(ASCEND310_GPIO_0_DIR, O_RDONLY);
            break;
        case GPIO1:
            fd_function = open(ASCEND310_GPIO_1_DIR, O_RDONLY);
            break;
        default:
            ERROR_LOG("wrong gpio number of ascend310!");
            return -1;
    }

    if (-1==fd_function)
    {
        ERROR_LOG("open gpio DIR file error gpio=%d", gpio);
        return -1;
    }

    if (-1==read(fd_function, direction, len))
    {
        ERROR_LOG("gpio read operation error gpio=%d", gpio);
        close(fd_function);
        return -1;
    }
    close(fd_function);
    if (direction[index_0]=='i' && direction[index_1]=='n')
    {
        return 0;
    }
    else if (direction[index_0]=='o' && direction[index_1]=='u' && direction[index_2]=='t')
    {
        return 1;
    }
    else
    {
        ERROR_LOG("invalid direction value!");
        return -1;
    }
}

int gpio_function(int gpio)
{
    if (gpio==GPIO0 || gpio==GPIO1)
    {
        return ascend310_gpio_function(gpio);
    }
    else if (gpio>=GPIO3 && gpio<=GPIO7)
    {
        return pca6416_gpio_function(gpio);
    }
    else
    {
        ERROR_LOG("invalid gpio number!");
        return -1;
    }
}

int cleanup(int gpio)
{
    ifsetup[gpio] = 0;
    if (-1 == setup(gpio, INPUT))
    {
        ERROR_LOG("cleanup error while gpio=%d!", gpio);
        return -1;
    }
    ifsetup[gpio] = 0;
    return 0;
}

int cleanup_all(void)
{
    int res;
    int gpio_num = 7;
    int gpio2 = 2;

    for (int i = 0; i <= gpio_num; i++)
    {
        if (i==gpio2)
        {
            continue;
        }
        ifsetup[i] = 0;
        res = setup(i, INPUT);
        if (-1==res)
        {
            ERROR_LOG("cleanup error while gpio=%d!", i);
            return -1;
        }
        ifsetup[i] = 0;
    }
    return 0;
}

int setwarnings(int status)
{
    if (status==0 || status==1)
    {
        EnableWarnings = status;
        return 0;
    }
    else
    {
        ERROR_LOG("wrong status! please input 0 or 1.");
        return -1;
    }
}