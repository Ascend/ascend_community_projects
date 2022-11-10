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

/* I2C Device */
#define I2C1_DEV_NAME                      "/dev/i2c-1"

#define I2C_RETRIES                        0x0701
#define I2C_TIMEOUT                        0x0702
#define I2C_SLAVE                          0x0703
#define I2C_RDWR                           0x0707
#define I2C_BUS_MODE                       0x0780
#define I2C_M_RD                           0x01
#define PCA6416_SLAVE_ADDR                 0x20
#define PCA6416_GPIO_CFG_REG               0x07
#define PCA6416_GPIO_PORARITY_REG          0x05
#define PCA6416_GPIO_OUT_REG               0x03
#define PCA6416_GPIO_IN_REG                0x01

#define ASCEND310_GPIO_0_DIR          "/sys/class/gpio/gpio504/direction"
#define ASCEND310_GPIO_1_DIR          "/sys/class/gpio/gpio444/direction"
#define ASCEND310_GPIO_0_VAL          "/sys/class/gpio/gpio504/value"
#define ASCEND310_GPIO_1_VAL          "/sys/class/gpio/gpio444/value"

#define GPIO0 0
#define GPIO1 1
#define GPIO3 3
#define GPIO4 4
#define GPIO5 5
#define GPIO6 6
#define GPIO7 7

/* GPIO MASK */
#define GPIO3_MASK                         0x10
#define GPIO4_MASK                         0x20
#define GPIO5_MASK                         0x40
#define GPIO6_MASK                         0x80
#define GPIO7_MASK                         0x08

#define INPUT  0
#define OUTPUT 1

#define HIGH 1
#define LOW  0

#define I2C_WRITE_MSG_LEN 2
#define I2C_READ_MSG_NUMS 2

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) if (EnableWarnings) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

struct i2c_msg {
    unsigned short addr;     /* slave address */
    unsigned short flags;
    unsigned short len;
    unsigned char *buf;     /* message data pointer */
};

struct i2c_rdwr_ioctl_data {
    struct i2c_msg *msgs;   /* i2c_msg[] pointer */
    int nmsgs;              /* i2c_msg Nums */
};

int fd_gpio;
int EnableWarnings;
int ifsetup[8];

int i2c_write(unsigned char slave, unsigned char reg, unsigned char value);
int i2c_read(unsigned char slave, unsigned char reg, unsigned char *buf);
int gpio_init(void);
int gpio_close(void);
int pca6416_setup(int gpio, int direction);
int ascend310_setup(int gpio, int direction);
int setup(int gpio, int direction);
int pca6416_output(int gpio, int value);
int ascend310_output(int gpio, int value);
int output(int gpio, int value);
int pca6416_input(int gpio);
int ascend310_input(int gpio);
int input(int gpio);
int pca6416_gpio_function(int gpio);
int ascend310_gpio_function(int gpio);
int gpio_function(int gpio);  // 0 for INPUT, 1 for OUTPUT
int cleanup(int gpio);  // cleanup one
int cleanup_all(void);  // cleanup all
int setwarnings(int status);