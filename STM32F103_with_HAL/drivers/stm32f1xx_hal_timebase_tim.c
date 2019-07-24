#include "stm32f1xx_hal.h"

TIM_HandleTypeDef TimHandle;
void TIM2_IRQHandler(void);

/*
 * This function configures the TIM2 as a time base source.
 * 	 The time source is configured to have 1ms time base with a dedicated
 * 	 Tick interrupt priority.
 * This function is called automatically at the beginning of program after
 * 	reset by HAL_Init() or at any time when clock is configured, by HAL_RCC_ClockConfig().
 */
HAL_StatusTypeDef HAL_InitTick(uint32_t TickPriority)
{
	RCC_ClkInitTypeDef		clkconfig;
	uint32_t							uwTimclock, uwAPB1Prescaler = 0U;
	uint32_t							uwPrescalerValue = 0U;
	uint32_t							pFLatency;

	/*Configure the TIM2 IRQ priority */
	HAL_NVIC_SetPriority(TIM2_IRQn, TickPriority, 0U);

	/* Enable the TIM2 global Interrupt */
	HAL_NVIC_EnableIRQ(TIM2_IRQn);

	/* Enable TIM2 clock */
	__HAL_RCC_TIM2_CLK_ENABLE();

	/* Get clock configuration */
	HAL_RCC_GetClockConfig(&clkconfig, &pFLatency);

	/* Get APB1 prescaler */
	uwAPB1Prescaler = clkconfig.APB1CLKDivider;

	/* Compute TIM2 clock */
	if (uwAPB1Prescaler == RCC_HCLK_DIV1)
	{
		uwTimclock = HAL_RCC_GetPCLK1Freq();
	}
	else
	{
		uwTimclock = 2 * HAL_RCC_GetPCLK1Freq();
	}

	/* Compute the prescaler value to have TIM2 counter clock equal to 1MHz */
	uwPrescalerValue = (uint32_t)((uwTimclock / 1000000U) - 1U);

	/* Initialize TIM2 */
	TimHandle.Instance = TIM2;

	/* Initialize TIMx peripheral as follow:
	 + Period = [(TIM2CLK/1000) - 1]. to have a (1/1000) s time base.
	 + Prescaler = (uwTimclock/1000000 - 1) to have a 1MHz counter clock.
	 + ClockDivision = 0
	 + Counter direction = Up
	*/
	TimHandle.Init.Period = (1000000U / 1000U) - 1U;
	TimHandle.Init.Prescaler = uwPrescalerValue;
	TimHandle.Init.ClockDivision = 0U;
	TimHandle.Init.CounterMode = TIM_COUNTERMODE_UP;
	TimHandle.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
	if (HAL_TIM_Base_Init(&TimHandle) == HAL_OK)
	{
		/* Start the TIM time Base generation in interrupt mode */
		return HAL_TIM_Base_Start_IT(&TimHandle);
	}

	return HAL_ERROR;
}

/*
 * Suspend Tick increment.
 * Disable the tick increment by disabling TIM2 update interrupt.
 */
void HAL_SuspendTick(void)
{
	__HAL_TIM_DISABLE_IT(&TimHandle, TIM_IT_UPDATE);
}

/*
 * Resume Tick increment.
 * Enable the tick increment by Enabling TIM2 update interrupt.
 */
void HAL_ResumeTick(void)
{
	__HAL_TIM_ENABLE_IT(&TimHandle, TIM_IT_UPDATE);
}

/*
 * Period elapsed callback in non blocking mode
 * This function is called when TIM2 interrupt took place, inside
 * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
 * a global variable "uwTick" used as application time base.
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
	(void) htim;
	HAL_IncTick();
}

void TIM2_IRQHandler(void)
{
	HAL_TIM_IRQHandler(&TimHandle);
}
