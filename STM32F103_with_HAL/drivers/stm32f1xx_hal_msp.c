#include "stm32f1xx_hal.h"

void HAL_MspInit(void)
{
	HAL_NVIC_EnableIRQ(SysTick_IRQn);
	HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_0);
	HAL_NVIC_SetPriority(SysTick_IRQn, 0, 0);
}

void HAL_MspDeInit(void)
{
}

void HAL_PPP_MspInit(void)
{
}

void HAL_PPP_MspDeInit(void)
{
}
