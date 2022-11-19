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
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/time.h>
#include <errno.h>
#include "smbus.h"

int i2c_2_init(void)
{
  // open i2c-2 device
  fd_smbus = open(I2C2_DEV_NAME, O_RDWR);
  if (fd_smbus < 0) {
    ERROR_LOG("can't open i2c-2!");
    return -1;
  }
  // set i2c-2 retries time
  if (ioctl(fd_smbus, I2C_RETRIES, 1) < 0) {
    close(fd_smbus);
    fd_smbus = 0;
    ERROR_LOG("set i2c-2 retry fail!");
    return -1;
  }
  // set i2c-2 timeout time, 10ms as unit
  if (ioctl(fd_smbus, I2C_TIMEOUT, 1) < 0) {
    close(fd_smbus);
    fd_smbus = 0;
    ERROR_LOG("set i2c-2 timeout fail!");
    return -1;
  }
  return 0;
}

int i2c_2_close(void)
{
  int error;
  enable_pec(0);
  error = close(fd_smbus);
  if (error < 0)
  {
    ERROR_LOG("can't close i2c-2!");
    return -1;
  }
  fd_smbus = 0;
  return 0;
}

int set_address(int slave)  // for smbus mode
{
	if (ioctl(fd_smbus, I2C_SLAVE, slave) < 0)
	{
		ERROR_LOG("fail ioctl I2C_SLAVE");
		return -1;
	}
	return 0;
}

int enable_pec(int status) // enable/disable PEC
{
	if (ioctl(fd_smbus, I2C_PEC, status) < 0)
	{
		ERROR_LOG("fail ioctl I2C_PEC");
		return -1;
	}
	return 0;
}

int get_pec(void)
{
	int pec = 0;
	if (ioctl(fd_smbus, I2C_PEC, pec) < 0)
	{
		ERROR_LOG("fail ioctl I2C_PEC");
		return -1;
	}
	return pec;
}

int get_funcs(void)
{
    int f = 0;
    if (ioctl(fd_smbus, I2C_FUNCS, f) < 0)
    {
        ERROR_LOG("fail ioctl I2C_FUNCS");
        return -1;
    }
    return f;
}

int i2c_ioctl_data_create(int file, char read_write, unsigned char command, int size, union i2c_smbus_data *data)
{
	struct i2c_smbus_ioctl_data args;
  
	args.read_write = read_write;
	args.command = command;
	args.size = size;
	args.data = data;

	int ret = ioctl(fd_smbus, I2C_SMBUS, &args);
  if (ret) {
    ERROR_LOG("fail ioctl I2C_SMBUS");
  }
	return ret;
}

int read_byte(int address)
{
    if (-1==set_address(address))
    {
        return -1;
    }
    union i2c_smbus_data data;
    if (i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_READ, 0, I2C_SMBUS_BYTE, &data))
    {
        return -1;
    }
    else
    {
        return 0x0FF & data.byte;
    }
}

int write_byte(int address, unsigned char value)
{
    if (-1==set_address(address))
    {
        return -1;
    }
    return i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, value, I2C_SMBUS_BYTE, NULL);
}

int read_byte_data(int address, unsigned char command)
{
    if (-1==set_address(address))
    {
        return -1;
    }
	union i2c_smbus_data data;
	if (i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_READ, command, I2C_SMBUS_BYTE_DATA, &data))
    {
		return -1;
    }
	else
    {
		return 0x0FF & data.byte;
    }
}

int write_byte_data(int address, unsigned char command, unsigned char value)
{
    if (-1==set_address(address))
    {
        return -1;
    }
	union i2c_smbus_data data;
	data.byte = value;
	return i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, command, I2C_SMBUS_BYTE_DATA, &data);
}

int read_word_data(int address, unsigned char command)
{
    if (-1==set_address(address))
    {
        return -1;
    }
	union i2c_smbus_data data;
	if (i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_READ, command, I2C_SMBUS_WORD_DATA, &data))
    {
		return -1;
    }
	else
    {
		return 0x0FFFF & data.word;
    }
}

int write_word_data(int address, unsigned char command, unsigned short value)
{
    if (-1 == set_address(address))
    {
        return -1;
    }
	union i2c_smbus_data data;
	data.word = value;
	return i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, command, I2C_SMBUS_WORD_DATA, &data);
}

unsigned char *read_block_data(int address, unsigned char command)
{
    unsigned char *values = (unsigned char*)malloc(sizeof(unsigned char) * I2C_SMBUS_BLOCK_MAX);
    memset(values, 0, sizeof(values));
    for(int i=0; i<I2C_SMBUS_BLOCK_MAX; i++)
        values[i] = read_byte_data(address, (command+i));
		return values;
}

int write_block_data(int address, unsigned char command, unsigned char *values)
{
    if (-1==set_address(address))
    {
        return -1;
    }
    int length = strlen(values) - 1;
    if (length > I2C_SMBUS_BLOCK_MAX)
    {
        ERROR_LOG("data length cannot exceed %d bytes", I2C_SMBUS_BLOCK_MAX);
        return -1;
    }
    union i2c_smbus_data data;
    for (int i = 1; i <= length; i++)
    {
        data.block[i] = values[i-1];
    }
    data.block[0] = length;
    return i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, command, I2C_SMBUS_BLOCK_DATA, &data);
}

int write_quick(int address)
{
    if (-1==set_address(address))
    {
      return -1;
    }
	return i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, 0, I2C_SMBUS_QUICK, NULL);
}

int process_call(int address, unsigned char command, unsigned short value)
{
    if (-1==set_address(address))
    {
        return -1;
    }
	union i2c_smbus_data data;
	data.word = value;
	if (i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, command, I2C_SMBUS_PROC_CALL, &data))
    {
		return -1;
    }
	else
	{
        if (i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_READ, command, I2C_SMBUS_PROC_CALL, &data))
        {
            return -1;
        }
        else
        {
            return 0x0FFFF & data.word;
        }
    }
}

unsigned char *read_i2c_block_data(int address, unsigned char command, unsigned char length)
{
    if (length > I2C_SMBUS_BLOCK_MAX)
    {
        ERROR_LOG("data length cannot exceed %d bytes", I2C_SMBUS_BLOCK_MAX);
        return NULL;
    }
    unsigned char *values = (unsigned char*)malloc(sizeof(unsigned char) * length);
    memset(values, 0, sizeof(values));
    for(int i=0; i<length; i++)
        values[i] = read_byte_data(address, (command+i));
		return values;
}

int write_i2c_block_data(int address, unsigned char command, unsigned char *values)
{
    if (-1==set_address(address))
    {
        return -1;
    }
    int length = strlen(values) - 1;
    if (length > I2C_SMBUS_BLOCK_MAX)
    {
        ERROR_LOG("data length cannot exceed %d bytes", I2C_SMBUS_BLOCK_MAX);
        return -1;
    }
    union i2c_smbus_data data;
    for (int i = 1; i <= length; i++)
    {
        data.block[i] = values[i-1];
    }
    data.block[0] = length;
    return i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, command, I2C_SMBUS_I2C_BLOCK_DATA, &data);
}

unsigned char *block_process_call(int address, unsigned char command, unsigned char *values)
{
    if (-1 == set_address(address))
    {
        return NULL;
    }
    unsigned char length = sizeof(values);
    if (length > I2C_SMBUS_BLOCK_MAX)
	{
        ERROR_LOG("data length cannot exceed %d bytes", I2C_SMBUS_BLOCK_MAX);
        return NULL;
    }
	union i2c_smbus_data data;
	int i;
	for (i = 1; i <= length; i++)
    {
		data.block[i] = values[i-1];
    }
	data.block[0] = length;
	if (i2c_ioctl_data_create(fd_smbus, I2C_SMBUS_WRITE, command, I2C_SMBUS_BLOCK_PROC_CALL, &data))
    {
		return NULL;
    }
	else
    {
		for (i = 1; i <= length; i++)
    {
		values[i-1] = data.block[i];
    }
		return values;
	}
}