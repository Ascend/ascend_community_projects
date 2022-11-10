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

#define I2C2_DEV_NAME "/dev/i2c-2"

#define I2C_RETRIES	0x0701 /* number of times a device address */
#define I2C_TIMEOUT	0x0702 /* set timeout - call with int */
// Commands from uapi/linux/i2c-dev.h
#define I2C_SLAVE 0x0703  // Use this slave address
#define I2C_SLAVE_FORCE 0x0706  // Use this slave address, even if it is already in use by a driver!
#define I2C_FUNCS 0x0705  // Get the adapter functionality mask
#define I2C_RDWR 0x0707  // Combined R/W transfer (one STOP only)
#define I2C_SMBUS 0x0720  // SMBus transfer. Takes pointer to i2c_smbus_ioctl_data
#define I2C_PEC 0x0708  // != 0 to use PEC with SMBus
#define I2C_SMBUS 0x0720 /* SMBus-level access */

// SMBus transfer read or write markers from uapi/linux/i2c.h
#define I2C_SMBUS_WRITE 0
#define I2C_SMBUS_READ 1

// Size identifiers uapi/linux/i2c.h
#define I2C_SMBUS_QUICK 0
#define I2C_SMBUS_BYTE 1
#define I2C_SMBUS_BYTE_DATA 2
#define I2C_SMBUS_WORD_DATA 3
#define I2C_SMBUS_PROC_CALL 4
#define I2C_SMBUS_BLOCK_DATA 5  // This isn't supported by Pure-I2C drivers with SMBUS emulation, like those in RaspberryPi, OrangePi, etc :(
#define I2C_SMBUS_BLOCK_PROC_CALL 7  // Like I2C_SMBUS_BLOCK_DATA, it isn't supported by Pure-I2C drivers either.
#define I2C_SMBUS_I2C_BLOCK_DATA 8
#define I2C_SMBUS_BLOCK_MAX 32

// i2c_msg flags from uapi/linux/i2c.h
#define I2C_M_RD 0x0001

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

// I2C_FUNCS
#define I2C 0x00000001
#define ADDR_10BIT 0x00000002
#define PROTOCOL_MANGLING 0x00000004  // I2C_M_IGNORE_NAK etc.
#define SMBUS_PEC 0x00000008
#define NOSTART 0x00000010  // I2C_M_NOSTART
#define SLAVE 0x00000020
#define SMBUS_BLOCK_PROC_CALL 0x00008000  // SMBus 2.0
#define SMBUS_QUICK 0x00010000
#define SMBUS_READ_BYTE 0x00020000
#define SMBUS_WRITE_BYTE 0x00040000
#define SMBUS_READ_BYTE_DATA 0x00080000
#define SMBUS_WRITE_BYTE_DATA 0x00100000
#define SMBUS_READ_WORD_DATA 0x00200000
#define SMBUS_WRITE_WORD_DATA 0x00400000
#define SMBUS_PROC_CALL 0x00800000
#define SMBUS_READ_BLOCK_DATA 0x01000000
#define SMBUS_WRITE_BLOCK_DATA 0x02000000
#define SMBUS_READ_I2C_BLOCK 0x04000000  // I2C-like block xfer
#define SMBUS_WRITE_I2C_BLOCK 0x08000000  // w/ 1-byte reg. addr.
#define SMBUS_HOST_NOTIFY 0x10000000
#define SMBUS_BYTE 0x00060000
#define SMBUS_BYTE_DATA 0x00180000
#define SMBUS_WORD_DATA 0x00600000
#define SMBUS_BLOCK_DATA 0x03000000
#define SMBUS_I2C_BLOCK 0x0c000000
#define SMBUS_EMUL 0x0eff0008

int fd_smbus;

union i2c_smbus_data {
    unsigned char byte;
    unsigned short word;
    unsigned char block[I2C_SMBUS_BLOCK_MAX + 3]; /* block[0] is used for length */
                              /* one more for read length in block process call */
                                                    /* and one more for PEC */
};

struct i2c_smbus_ioctl_data {
    char read_write;
    unsigned char command;
    int size;
    union i2c_smbus_data *data;
};

int i2c_2_init(void);
int i2c_2_close(void);
int set_address(int slave);
int enable_pec(int status);
int get_pec(void);
int get_funcs(void);
int i2c_ioctl_data_create(int file, char read_write, unsigned char command, int size, union i2c_smbus_data *data);
int read_byte(int address);
int write_byte(int address, unsigned char value);
int read_byte_data(int address, unsigned char command);
int write_byte_data(int address, unsigned char command, unsigned char value);
int read_word_data(int address, unsigned char command);
int write_word_data(int address, unsigned char command, unsigned short value);
unsigned char *read_block_data(int address, unsigned char command);
int write_block_data(int address, unsigned char command, unsigned char *values);
int write_quick(int address);
int process_call(int address, unsigned char command, unsigned short value);
unsigned char *read_i2c_block_data(int address, unsigned char command, unsigned char length);
int write_i2c_block_data(int address, unsigned char command, unsigned char *values);
unsigned char *block_process_call(int address, unsigned char command, unsigned char *values);