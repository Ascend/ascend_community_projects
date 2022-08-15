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
	short resive_data[6]; // 保存收到的 mpu6050转换结果数据，依次为 AX(x轴角度), AY, AZ 。GX(x轴加速度), GY ,GZ
    int index_0 = 0, index_1 = 1, index_2 = 2, index_3 = 3, index_4 = 4, index_5 = 5;
	int error;
	int AX, AY, AZ, GX, GY, GZ; // 原始数据
	float AX_zeroset = -0.05, AY_zeroset = 0.00, AZ_zeroset = 0.19, GX_zeroset = 75.93, GY_zeroset = 19.12, GZ_zeroset = 21.35; // 调零参数
	float acc_trans = 16384, gyro_trans = 16.4;
	float AX_act, AY_act, AZ_act, GX_act, GY_act, GZ_act;
	char *buf;
	int sleep_time = 100000;

	/* 打开文件 */
	int fd = open("/dev/I2C1_mpu6050", O_RDWR);
	if(fd<0)
	{
		printf("open file : %s failed !\n", argv[0]);
		return -1;
	}

	while(1) {
	/* 读取数据 */
	error = read(fd, resive_data, 12);
	if(error<0)
	{
		printf("write file error! \n");
		close(fd);
		/* 判断是否关闭成功 */
	}

	AX = (int)resive_data[index_0];
	AY = (int)resive_data[index_1];
	AZ = (int)resive_data[index_2];
	GX = (int)resive_data[index_3];
	GY = (int)resive_data[index_4];
	GZ = (int)resive_data[index_5];

	/* 调零 单位转换 */
	AX_act = (float)(AX) / acc_trans + AX_zeroset;
	AY_act = (float)(AY) / acc_trans + AY_zeroset;
	AZ_act = (float)(AZ) / acc_trans + AZ_zeroset;
	GX_act = ((float)(GX) + GX_zeroset) / gyro_trans;
	GY_act = ((float)(GY) + GY_zeroset) / gyro_trans;
	GZ_act = ((float)(GZ) + GZ_zeroset) / gyro_trans;

	/* 将解算后的数据显示在LCD上 */
	sprintf(buf, "bash display_mpu6050.sh %.2f %.2f %.2f %.2f %.2f %.2f", AX_act, AY_act, AZ_act, GX_act, GY_act, GZ_act);
	system(buf);

	usleep(sleep_time);
	}

	/* 关闭文件 */
	error = close(fd);
	if(error<0)
	{
		printf("close file error! \n");
	}

	return 0;
}
