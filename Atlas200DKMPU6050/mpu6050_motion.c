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
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char *argv[])
{
	int fd, fd_uart;
	char* filename = "/dev/I2C1_mpu6050";
	char* uartname = "/dev/ttyAMA1";
	int resive_size = 12;
	int error;
	int index_0 = 0, index_1 = 1, index_2 = 2, index_3 = 3, index_4 = 4, index_5 = 5;
	char* data;
	int sleep_time = 100000;
	short resive_data[6];  // 保存收到的 mpu6050转换结果数据，依次为 AX(x轴角度), AY, AZ 。GX(x轴加速度), GY ,GZ
	int accel_x_adc, accel_y_adc, accel_z_adc, gyro_x_adc, gyro_y_adc, gyro_z_adc;
	float accel_x_act, accel_y_act, accel_z_act, gyro_x_act, gyro_y_act, gyro_z_act;
	float gyro_x_act_last = 0, gyro_y_act_last = 0, gyro_z_act_last = 0;
	float accel_x_zeroset = -0.05, accel_y_zeroset = 0.01, accel_z_zeroset = 0.19, gyro_x_zeroset = 75.93, gyro_y_zeroset = 19.12, gyro_z_zeroset = 21.35; // 调零参数,根据设备调整
	float accel_trans = 16384, gyro_trans = 16.4; // 转换系数
	float a_pitch = 0, a_roll = 0; // 加速度计解算出的姿态
	float g_pitch = 0, g_yaw = 0, g_roll = 0; // 陀螺仪解算出的姿态
	float pitch = 0, yaw = 0, roll = 0; // 最终姿态
	float rad2deg = 57.30; // 弧度制转化角度制
	float ms2s = 1000000; // 毫秒转化秒
	float dt; // 积分时间长度
	float half = 0.5;
	clock_t now, last;
	float rateGA = 0.5;
	int uart_size = 30;

	/*打开文件*/
	fd = open(filename, O_RDWR);
	if(fd < 0)
	{
		printf("open file : %s failed !\n", argv[0]);
		return -1;
	}

	fd_uart = open(uartname, O_RDWR);
	if (fd_uart < 0)
	{
		printf("can't open file %s\r\n", uartname);
		return -1;
	}

	now = clock();
	while(1)
	{
		/* 读取数据 */
		error = read(fd, resive_data, resive_size);
		if(error < 0)
		{
			printf("write file error! \n");
			close(fd);
			return -1;
		}
		/* 原始数据 */
		accel_x_adc = (int)resive_data[index_0];
		accel_y_adc = (int)resive_data[index_1];
		accel_z_adc = (int)resive_data[index_2];
		gyro_x_adc = (int)resive_data[index_3];
		gyro_y_adc = (int)resive_data[index_4];
		gyro_z_adc = (int)resive_data[index_5];
		
		/* 数据转换 */
		accel_x_act = (float)(accel_x_adc) / accel_trans + accel_x_zeroset;
		accel_y_act = (float)(accel_y_adc) / accel_trans + accel_y_zeroset;
		accel_z_act = (float)(accel_z_adc) / accel_trans + accel_z_zeroset;
		gyro_x_act = ((float)(gyro_x_adc) + gyro_x_zeroset) / gyro_trans;
		gyro_y_act = ((float)(gyro_y_adc) + gyro_y_zeroset) / gyro_trans;
		gyro_z_act = ((float)(gyro_z_adc) + gyro_z_zeroset) / gyro_trans;

		/* 利用加速度值姿态解算 */
		a_roll = atan(accel_y_act / accel_z_act) * rad2deg;
		a_pitch = -atan(accel_x_act / sqrt(accel_y_act * accel_y_act + accel_z_act * accel_z_act)) * rad2deg;

		last = now;
		now = clock();
		dt = (float)(now - last) / CLOCKS_PER_SEC + (float)sleep_time / ms2s;

		/* 利用陀螺仪姿态解算 */
		g_pitch = pitch + (gyro_y_act + gyro_y_act_last) * dt * half;
		g_yaw = yaw + (gyro_z_act + gyro_z_act_last) * dt * half;
		g_roll = roll + (gyro_x_act + gyro_x_act_last) * dt * half;

		gyro_x_act_last = gyro_x_act;
		gyro_y_act_last = gyro_y_act;
		gyro_z_act_last = gyro_z_act;

		/* 数据融合 */
		pitch = g_pitch * rateGA + a_pitch * (1 - rateGA);
		yaw = g_yaw;
		roll = g_roll * rateGA + a_roll * (1 - rateGA);
		printf("pitch=%.2f,yaw=%.2f,roll=%.2f\n", pitch, yaw, roll);

		/* 利用串口传输数据 */
		sprintf(data, "%.2f,%.2f,%.2f\n", pitch, yaw, roll);
		error = write(fd_uart, data, uart_size);
		
		if(error < 0)
		{
			printf("write file error! \n");
			close(fd);
			return -1;
		}

		usleep(sleep_time);
	}
 	/*关闭文件*/
	error = close(fd);
	error += close(fd_uart);
	if(error < 0)
	{
 		printf("close file error! \n");
	}
	return 0;
}
