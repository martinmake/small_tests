#include <stm32f1xx.h>

#define LED1_GPIO_CLK_ENABLE() __HAL_RCC_GPIOA_CLK_ENABLE();
#define LED1_GPIO_PORT         GPIOA
#define LED1_PIN               GPIO_PIN_8
#define LED2_GPIO_CLK_ENABLE() __HAL_RCC_GPIOD_CLK_ENABLE();
#define LED2_GPIO_PORT         GPIOD
#define LED2_PIN               GPIO_PIN_2

void init_gpio(void)
{
	GPIO_InitTypeDef GPIO_InitStruct;
	GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull  = GPIO_PULLUP;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

	GPIO_InitStruct.Pin = LED1_PIN;
	HAL_GPIO_Init(LED1_GPIO_PORT, &GPIO_InitStruct);

	GPIO_InitStruct.Pin = LED2_PIN;
	HAL_GPIO_Init(LED2_GPIO_PORT, &GPIO_InitStruct);

	LED1_GPIO_CLK_ENABLE();
	LED2_GPIO_CLK_ENABLE();
}

// void SystemClock_Config(void);

int main(void)
{
	HAL_Init();
 	SystemClock_Config();
	init_gpio();

	// HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_0);
	// HAL_NVIC_SetPriority(SysTick_IRQn, 0, 0);
	// __enable_irq();

	while (1)
	{
		HAL_GPIO_WritePin(LED1_GPIO_PORT, LED1_PIN, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(LED2_GPIO_PORT, LED2_PIN, GPIO_PIN_SET);
		HAL_Delay(200);
		HAL_GPIO_WritePin(LED2_GPIO_PORT, LED2_PIN, GPIO_PIN_RESET);
		HAL_GPIO_WritePin(LED1_GPIO_PORT, LED1_PIN, GPIO_PIN_SET);
		HAL_Delay(200);
	}
}

/*
void SystemClock_Config(void)
{
	RCC_ClkInitTypeDef clkinitstruct = {0};
	RCC_OscInitTypeDef oscinitstruct = {0};

	// Configure PLL ------------------------------------------------------
	// PLL configuration: PLLCLK = (HSI / 2) * PLLMUL = (8 / 2) * 16 = 64 MHz
	// PREDIV1 configuration: PREDIV1CLK = PLLCLK / HSEPredivValue = 64 / 1 = 64 MHz
	// Enable HSI and activate PLL with HSi_DIV2 as source
	oscinitstruct.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
	oscinitstruct.HSEState            = RCC_HSE_OFF;
	oscinitstruct.LSEState            = RCC_LSE_OFF;
	oscinitstruct.HSIState            = RCC_HSI_ON;
	oscinitstruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
	oscinitstruct.HSEPredivValue      = RCC_HSE_PREDIV_DIV1;
	oscinitstruct.PLL.PLLState        = RCC_PLL_ON;
	oscinitstruct.PLL.PLLSource       = RCC_PLLSOURCE_HSI_DIV2;
	oscinitstruct.PLL.PLLMUL          = RCC_PLL_MUL16;
	if (HAL_RCC_OscConfig(&oscinitstruct)!= HAL_OK) while(1) {}

	// Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2 clocks dividers
	clkinitstruct.ClockType      = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
	clkinitstruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
	clkinitstruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
	clkinitstruct.APB2CLKDivider = RCC_HCLK_DIV1;
	clkinitstruct.APB1CLKDivider = RCC_HCLK_DIV2;
	if (HAL_RCC_ClockConfig(&clkinitstruct, FLASH_LATENCY_2)!= HAL_OK) while(1) {}
}
*/

#ifdef	USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) { while (1) {} }
#endif


