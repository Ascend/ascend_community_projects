# 基于昇腾A200DK适配GPIO/I2C外设库RPI.GPIO/smbus2

## 1 概述

### 1.1 概要描述

基于昇腾A200DK适配GPIO/I2C外设库RPI.GPIO/smbus2，GPIO接口与树莓派的GPIO操作模块RPI.GPIO保持一致，I2C操作接口与python的smbus2模块保持一致。

### 1.2 实现流程

    1、环境搭建：Atlask 200 DK
    2、用C实现RPi.GPIO/smbus2接口
    3、绑定到python接口
    4、用外设或示波器测试

## 2 方案介绍

### 2.1 Atlas 200 DK GPIO/I2C介绍

表2-1-1 GPIO相关参数：

| 名称 | 管脚编号 | gpio编号 |
| :---: | :---: | :---: |
| GPIO0 | 7  | 504 |
| GPIO1 | 11 | 444 |
| GPIO3 | 29 | 429 |
| GPIO4 | 31 | 430 |
| GPIO5 | 33 | 431 |
| GPIO6 | 35 | 432 |
| GPIO7 | 37 | 428 |

（注：GPIO3-7的gpio编号仅适用B板）

表2-1-2 I2C相关参数：

| 名称 | 管脚编号 |
| :---: | :---: |
| I2C2-SDA | 3 |
| I2C2-SCL | 5 |

### 2.2 RPI.GPIO模块介绍

(1)setup：设置GPIO方向，0为INPUT，1为OUTPUT

(2)cleanup：释放CPIO，即将GPIO方向设置为INPUT

(3)output：设置方向为OUTPUT的GPIO的电平，0为低电平，1为高电平

(4)input：读取GPIO的电平值，0为低电平，1为高电平

(5)gpio_function：读取GPIO的功能，0为INPUT，1为OUTPUT

(6)setwarnings：设置是否显示警告，0为关闭，1为开启

### 2.3 smbus2模块介绍

(1)get_funcs：读取I2C功能

(2)enable_pec：开关PEC，0为关闭，1为开启

(3)read_byte：读1个字节

(4)write_byte：写1个字节

(5)read_byte_data：从指定寄存器读1个字节

(6)write_byte_data：在指定寄存器中写1个字节

(7)read_word_data：从指定寄存器读一个词（2个字节）

(8)write_word_data：在指定寄存器中写一个词（2个字节）

(9)read_i2c_block_data：从指定寄存器读一个数据块（给定字节，不大于32个字节）

(10)write_i2c_block_data：在指定寄存器中写一个数据块（不大于32个字节）

(11)write_quick：执行快速传输，进行测试

(12)process_call：执行一个SMBus进程调用，发送一个16位的值并接收一个16位的响应

(13)read_block_data：从指定寄存器读一个数据块（32个字节）

(14)write_block_data：在指定寄存器中写一个数据块（不大于32个字节）

(15)block_process_call：执行一个SMBus块进程调用，发送一个可变大小的数据块并接收另一个可变大小的响应（不大于32个字节）

### 2.4 代码目录结构与说明

```
├──gpio
│   ├── a200dkgpio
│   │   └── __init__.py
│   ├── a200dkgpio.egg-info	# 由setup.py生成
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── src
│   │   ├── gpio.h
│   │   ├── gpio.c	# C实现
│   │   └── py_gpio.c	# 绑定到python接口
│   ├── test
│   └── setup.py		# 创建whl包和安装相应模块
├── smbus
│   ├── a200dksmbus
│   │   └── __init__.py
│   ├── a200dksmbus.egg-info	# 由setup.py生成
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── src
│   │   ├── smbus.h
│   │   ├── smbus.c	# C实现
│   │   └── py_smbus.c	# 绑定到python接口
│   ├── test
│   └── setup.py		# 创建whl包和安装相应模块
└──README.md
```

## 3 开发准备

### 3.1 环境依赖说明

| 软件名称 | 版本 |
| :---: | :---: |
|Ubuntu|18.04.1 LTS|
|GCC|7.5.0|
|Python|3.7.13|
|CANN|5.0.5alpha001|
|固件与驱动|1.0.12|

### 3.2 环境搭建

见Atlas200dk-MindXSDK 开发环境搭建一站式导航：
https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk%E5%BC%80%E5%8F%91%E6%9D%BF%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md

### 3.3 获取GPIO/I2C操作权限

1.su root，然后输入密码，切换成root用户。

2.vim /etc/rc.local 在exit0前增加如下指令：

    echo 504 >/sys/class/gpio/export
    echo 444 >/sys/class/gpio/export
    chown -R HwHiAiUser /sys/class/gpio/gpio444
    chown -R HwHiAiUser /sys/class/gpio/gpio504
    chown -R HwHiAiUser /sys/class/gpio/gpio444/direction
    chown -R HwHiAiUser /sys/class/gpio/gpio504/direction
    chown -R HwHiAiUser /sys/class/gpio/gpio444/value
    chown -R HwHiAiUser /sys/class/gpio/gpio504/value
    chown -R HwHiAiUser /dev/i2c-1
    chown -R HwHiAiUser /dev/i2c-2
    usermod -aG HwHiAiUser HwHiAiUser

3.重启运行环境。

## 4 开发流程

### 4.1 GPIO外设库

在开发者板上的GPIO都由40pin工具板引出。

直接从昇腾AI处理器引出的GPIO有：GPIO0、GPIO1。

由PCA6416引出的GPIO有：GPIO3、GPIO4、GPIO5、GPIO6、GPIO7。

GPIO0、GPIO1：通过读写内核态创建的设备节点文件直接去控制从昇腾AI处理器上引出的GPIO。
对应文件如下：

    GPIO0的方向：/sys/class/gpio/gpio504/direction (in 输入,out 输出)
    GPIO0的电平：/sys/class/gpio/gpio504/value (0 低电平,1 高电平)
    GPIO1的方向：/sys/class/gpio/gpio444/direction
    GPIO1的电平：/sys/class/gpio/gpio444/value

说明：GPIO0、GPIO1管脚作为输出管脚，必须外置上拉电阻增加驱动能力，建议上拉电阻的值为1K～10K。

GPIO3、GPIO4、GPIO5、GPIO6、GPIO7：通过在用户态编写PCA6416扩展GPIO的操作程序，通过控制I2C实现对PCA6416设备的读写，从而对PCA6416的扩展GPIO进行输入输出操作。

具体步骤如下：

    1.由于PCA6416连接的是I2C-1，打开I2C-1设备文件，获取文件描述符；
    2.通过I2C读写操作，读写PCA6416的寄存器，从而配置PCA6416扩展IO的输入输出特性，包括GPIO输入输出属性配置，GPIO输出电平配置，获取当前GPIO管脚电平状态等。
    3.操作完成后，关闭I2C-1设备。
    4.2I2C外设库

I2C2-SDA和I2C2-SCL组成I2C-2，可以用来外接传感器，与其他模块通信等，速率最高支持400KHz。
说明：I2C-1接口属于板内接口，不对外开放。

在用户态下通过I2C读写程序实现对I2C外围设备的读写操作具体步骤如下：

    1.打开 I2C 总线对应的设备文件，获取文件描述符。
    fd = open("/dev/i2c-2", O_RDWR);
    2.进行数据读写。

## 5 编译与运行

### 5.1 GPIO C示例步骤

**步骤1** 执行如下命令，进入到GPIO代码资源目录下。

    cd gpio/src

**步骤2** 执行如下命令，将smbus.c编译成.so文件。

    gcc gpio.c -fPIC -shared -o libgpio.so

**步骤3** 执行如下命令，在动态库配置文件ld.so.conf文件中添加路径：/usr/local/lib

    vi /etc/ld.so.conf

**步骤4** 执行如下命令，将libsmbus.so复制到路径/usr/local/lib下。

    cp libgpio.so /usr/local/lib

**步骤5** 执行如下命令，让动态链接库为系统所共享。

    ldconfig

**步骤6** 执行如下命令，将gpio.h复制到路径/usr/include下。。

    cp gpio.h /usr/include

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译测试代码。

    gcc c_test.c -L. -lgpio -o test

**步骤9** 执行如下命令，运行测试代码。

    ./test

### 5.2 GPIO Python示例步骤

**步骤1** 执行如下命令，创建一个python版本为3.7的conda环境py37。

    conda create -n py37 python=3.7

**步骤2** 执行如下命令，激活环境py37。

    conda activate py37

**步骤3** 执行如下命令，进入到GPIO项目目录下。

    cd gpio

**步骤4** 执行如下命令，运行setup.py，生成whl包。

    python setup.py bdist_wheel

**步骤5** 执行如下命令，进入到dist目录下。

    cd dist

**步骤6** 执行如下命令，安装whl包。

    pip install a200dkgpio-0.0.1-cp37-cp37m-linux_aarch64.whl

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译并运行测试代码。

    python py_test.py

### 5.3 I2C C示例步骤

**步骤1** 执行如下命令，进入到I2C代码资源目录下。

    cd smbus/src

**步骤2** 执行如下命令，将smbus.c编译成.so文件。

    gcc smbus.c -fPIC -shared -o libsmbus.so

**步骤3** 执行如下命令，在动态库配置文件ld.so.conf文件中添加路径：/usr/local/lib

    vi /etc/ld.so.conf

**步骤4** 执行如下命令，将libsmbus.so复制到路径/usr/local/lib下。

    cp libsmbus.so /usr/local/lib

**步骤5** 执行如下命令，让动态链接库为系统所共享。

    ldconfig

**步骤6** 执行如下命令，将gpio.h复制到路径/usr/include下。。

    cp smbus.h /usr/include

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译测试代码。

    gcc c_test.c -L. -lsmbus -o test

**步骤9** 执行如下命令，运行测试代码。

    ./test

### 5.4 I2C Python示例步骤

**步骤1** 执行如下命令，创建一个python版本为3.7的conda环境py37。

    conda create -n py37 python=3.7

**步骤2** 执行如下命令，激活环境py37。

    conda activate py37

**步骤3** 执行如下命令，进入到I2C项目目录下。

    cd smbus

**步骤4** 执行如下命令，运行setup.py，生成whl包。

    python setup.py bdist_wheel

**步骤5** 执行如下命令，进入到dist目录下。

    cd dist

**步骤6** 执行如下命令，安装whl包。

    pip install a200dksmbus-0.0.1-cp37-cp37m-linux_aarch64.whl

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译并运行测试代码。

    python py_test.py

## 6 参考链接

> Atlas200dk-MindXSDK 开发环境搭建一站式导航：
https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk%E5%BC%80%E5%8F%91%E6%9D%BF%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md

> Atlas 200 DK开发者套件 1.0.13 外围设备驱动接口操作指南：
https://www.hiascend.com/document/detail/zh/Atlas200DKDeveloperKit/1013/peripheralref/atlaspd_07_0001.html

> 树莓派的GPIO操作模块RPi.GPIO：https://pypi.org/project/RPi.GPIO/#files

> python的smbus2模块：https://pypi.org/project/smbus2/#files

> Atlas200DK各硬件接口的使用样例：
https://gitee.com/ascend/samples/tree/master/cplusplus/level1_single_api/5_200dk_peripheral