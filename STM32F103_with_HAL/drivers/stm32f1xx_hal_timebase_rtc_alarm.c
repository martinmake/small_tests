#include "stm32f1xx_hal.h"

/* Uncomment the line below to select the appropriate RTC Clock source for your application:
 + RTC_CLOCK_SOURCE_HSE: can be selected for applications requiring timing precision.
 + RTC_CLOCK_SOURCE_LSE: can be selected for applications with low constraint on timing precision.
 + RTC_CLOCK_SOURCE_LSI: can be selected for applications with low constraint on timing precision.
 */
#define RTC_CLOCK_SOURCE_HSE
// #define RTC_CLOCK_SOURCE_LSE
// #define RTC_CLOCK_SOURCE_LSI

RTC_HandleTypeDef hRTC_Handle;
void RTC_Alarm_IRQHandler(void);

/*
 * This function configures the RTC_ALARMA as a time base source.
 * The time source is configured	to have 1ms time base with a dedicated
 * Tick interrupt priority.
 * This function is called	automatically at the beginning of program after
 * reset by HAL_Init() or at any time when clock is configured, by HAL_RCC_ClockConfig().
 */
HAL_StatusTypeDef HAL_InitTick(uint32_t TickPriority)
{
	__IO uint32_t counter = 0U;

	RCC_OscInitTypeDef				RCC_OscInitStruct;
	RCC_PeriphCLKInitTypeDef	PeriphClkInitStruct;

#ifdef RTC_CLOCK_SOURCE_LSE
	/* Configue LSE as RTC clock soucre */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSE;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.LSEState = RCC_LSE_ON;
	PeriphClkInitStruct.RTCClockSelection = RCC_RTCCLKSOURCE_LSE;
#elif defined (RTC_CLOCK_SOURCE_LSI)
	/* Configue LSI as RTC clock soucre */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSI;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.LSIState = RCC_LSI_ON;
	PeriphClkInitStruct.RTCClockSelection = RCC_RTCCLKSOURCE_LSI;
#elif defined (RTC_CLOCK_SOURCE_HSE)
	/* Configue HSE as RTC clock soucre */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	PeriphClkInitStruct.RTCClockSelection = RCC_RTCCLKSOURCE_HSE_DIV128;
#else
#error Please select the RTC Clock source
#endif /* RTC_CLOCK_SOURCE_LSE */

	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) == HAL_OK)
	{
		PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_RTC;
		if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) == HAL_OK)
		{
			/* Enable RTC Clock */
			__HAL_RCC_RTC_ENABLE();

			hRTC_Handle.Instance = RTC;
			/* Configure RTC time base to 10Khz */
			hRTC_Handle.Init.AsynchPrediv = (HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_RTC) / 10000) - 1;
			hRTC_Handle.Init.OutPut = RTC_OUTPUTSOURCE_NONE;
			HAL_RTC_Init(&hRTC_Handle);

			/* Disable the write protection for RTC registers */
			__HAL_RTC_WRITEPROTECTION_DISABLE(&hRTC_Handle);

			/* Clear flag alarm A */
			__HAL_RTC_ALARM_CLEAR_FLAG(&hRTC_Handle, RTC_FLAG_ALRAF);

			counter = 0U;
			/* Wait till RTC ALRAF flag is set and if Time out is reached exit */
			while (__HAL_RTC_ALARM_GET_FLAG(&hRTC_Handle, RTC_FLAG_ALRAF) != RESET)
			{
				if (counter++ == SystemCoreClock / 48U) /* Timeout = ~ 1s */
				{
					return HAL_ERROR;
				}
			}

			/* Set RTC COUNTER MSB word */
			hRTC_Handle.Instance->ALRH = 0x00U;
			/* Set RTC COUNTER LSB word */
			hRTC_Handle.Instance->ALRL = 0x09U;

			/* RTC Alarm Interrupt Configuration: EXTI configuration */
			__HAL_RTC_ALARM_EXTI_ENABLE_IT();
			__HAL_RTC_ALARM_EXTI_ENABLE_RISING_EDGE();

			/* Clear Second and overflow flags */
			CLEAR_BIT(hRTC_Handle.Instance->CRL, (RTC_FLAG_SEC | RTC_FLAG_OW));

			/* Set RTC COUNTER MSB word */
			hRTC_Handle.Instance->CNTH = 0x00U;
			/* Set RTC COUNTER LSB word */
			hRTC_Handle.Instance->CNTL = 0x00U;

			/* Configure the Alarm interrupt */
			__HAL_RTC_ALARM_ENABLE_IT(&hRTC_Handle, RTC_IT_ALRA);

			/* Enable the write protection for RTC registers */
			__HAL_RTC_WRITEPROTECTION_ENABLE(&hRTC_Handle);

			/* Wait till RTC is in INIT state and if Time out is reached exit */
			counter = 0U;
			while ((hRTC_Handle.Instance->CRL & RTC_CRL_RTOFF) == (uint32_t)RESET)
			{
				if (counter++ == SystemCoreClock / 48U) /* Timeout = ~ 1s */
				{
					return HAL_ERROR;
				}
			}

			HAL_NVIC_SetPriority(RTC_Alarm_IRQn, TickPriority, 0U);
			HAL_NVIC_EnableIRQ(RTC_Alarm_IRQn);
			return HAL_OK;
		}
	}
	return HAL_ERROR;
}

/*
 * Suspend Tick increment.
 * Disable the tick increment by disabling RTC ALARM interrupt.
 */
void HAL_SuspendTick(void)
{
	/* Disable RTC ALARM update Interrupt */
	__HAL_RTC_ALARM_DISABLE_IT(&hRTC_Handle, RTC_IT_ALRA);
}

/*
 * Resume Tick increment.
 * Enable the tick increment by Enabling RTC ALARM interrupt.
 */
void HAL_ResumeTick(void)
{
	__IO uint32_t counter = 0U;

	/* Disable the write protection for RTC registers */
	__HAL_RTC_WRITEPROTECTION_DISABLE(&hRTC_Handle);

	/* Set RTC COUNTER MSB word */
	hRTC_Handle.Instance->CNTH = 0x00U;
	/* Set RTC COUNTER LSB word */
	hRTC_Handle.Instance->CNTL = 0x00U;

	/* Clear Second and overflow flags */
	CLEAR_BIT(hRTC_Handle.Instance->CRL, (RTC_FLAG_SEC | RTC_FLAG_OW | RTC_FLAG_ALRAF));

	/* Enable RTC ALARM Update interrupt */
	__HAL_RTC_ALARM_ENABLE_IT(&hRTC_Handle, RTC_IT_ALRA);

	/* Enable the write protection for RTC registers */
	__HAL_RTC_WRITEPROTECTION_ENABLE(&hRTC_Handle);

	/* Wait till RTC is in INIT state and if Time out is reached exit */
	while ((hRTC_Handle.Instance->CRL & RTC_CRL_RTOFF) == (uint32_t)RESET)
	{
		if (counter++ == SystemCoreClock / 48U) /* Timeout = ~ 1s */
		{
			break;
		}
	}
}

/*
 * ALARM A Event Callback in non blocking mode
 * This function is called	when RTC_ALARM interrupt took place, inside
 * RTC_ALARM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
 * a global variable "uwTick" used as application time base.
 */
void HAL_RTC_AlarmAEventCallback(RTC_HandleTypeDef *hrtc)
{
	__IO uint32_t counter = 0U;

	HAL_IncTick();

	__HAL_RTC_WRITEPROTECTION_DISABLE(hrtc);

	/* Set RTC COUNTER MSB word */
	WRITE_REG(hrtc->Instance->CNTH, 0x00U);
	/* Set RTC COUNTER LSB word */
	WRITE_REG(hrtc->Instance->CNTL, 0x00U);

	/* Clear Second and overflow flags */
	CLEAR_BIT(hrtc->Instance->CRL, (RTC_FLAG_SEC | RTC_FLAG_OW));

	/* Enable the write protection for RTC registers */
	__HAL_RTC_WRITEPROTECTION_ENABLE(hrtc);

	/* Wait till RTC is in INIT state and if Time out is reached exit */
	while ((hrtc->Instance->CRL & RTC_CRL_RTOFF) == (uint32_t)RESET)
	{
		if (counter++ == SystemCoreClock / 48U) /* Timeout = ~ 1s */
		{
			break;
		}
	}
}

/*
 * This function handles RTC ALARM interrupt request.
 */
void RTC_Alarm_IRQHandler(void)
{
	HAL_RTC_AlarmIRQHandler(&hRTC_Handle);
}
