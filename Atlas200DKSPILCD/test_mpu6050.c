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

int main(int argc, char *argv[])
{
	int fd;
	short buffer[6];
	int index_0 = 0, index_1 = 1, index_2 = 2, index_3 = 3, index_4 = 4, index_5 = 5;
	char *display_cmd;
	int sleep_time = 100000;

	int accel_x_adc, accel_y_adc, accel_z_adc, gyro_x_adc, gyro_y_adc, gyro_z_adc;
	float accel_x_act, accel_y_act, accel_z_act, gyro_x_act, gyro_y_act, gyro_z_act;
	float accel_x_zeroset = -0.05, accel_y_zeroset = 0.01, accel_z_zeroset = 0.19, gyro_x_zeroset = 75.93, gyro_y_zeroset = 19.12, gyro_z_zeroset = 21.35; // 调零参数,根据设备调整
	float accel_trans = 16384, gyro_trans = 16.4; // 转换系数

	int ret = 0;
	char* filename = "/dev/mpu6050";
	fd = open(filename, O_RDWR);
	if (fd < 0) {
		printf("can't open file %s\r\n", filename);
		return -1;
	}

	while (1) {
		ret = read(fd, buffer, sizeof(buffer));
		if (ret == 0) {
			/* 原始数据 */
			accel_x_adc = (int)buffer[index_3];
			accel_y_adc = (int)buffer[index_4];
			accel_z_adc = (int)buffer[index_5];
			gyro_x_adc = (int)buffer[index_0];
			gyro_y_adc = (int)buffer[index_1];
			gyro_z_adc = (int)buffer[index_2];

			/* 数据转换 */
			accel_x_act = (float)(accel_x_adc) / accel_trans + accel_x_zeroset;
			accel_y_act = (float)(accel_y_adc) / accel_trans + accel_y_zeroset;
			accel_z_act = (float)(accel_z_adc) / accel_trans + accel_z_zeroset;
			gyro_x_act = ((float)(gyro_x_adc) + gyro_x_zeroset) / gyro_trans;
			gyro_y_act = ((float)(gyro_y_adc) + gyro_y_zeroset) / gyro_trans;
			gyro_z_act = ((float)(gyro_z_adc) + gyro_z_zeroset) / gyro_trans;

			/* 将数据显示在LCD上 */
			sprintf(display_cmd, "bash display_mpu6050.sh %.2f %.2f %.2f %.2f %.2f %.2f", accel_x_act, accel_y_act, accel_z_act, gyro_x_act, gyro_y_act, gyro_z_act);
			system(display_cmd);
		}
		usleep(sleep_time);
	}
	close(fd);
	return 0;
}
