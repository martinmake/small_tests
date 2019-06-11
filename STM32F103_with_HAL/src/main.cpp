#include <stm32f1xx.h>

#define LED_GPIO_PORT GPIOA
#define LED_PIN       GPIO_PIN_8
// #define LED_GPIO_PORT GPIOD
// #define LED_PIN       GPIO_PIN_2

void init(void)
{
	HAL_Init();

	__enable_irq();
	__HAL_RCC_GPIOA_CLK_ENABLE();

	GPIO_InitTypeDef GPIO_InitStruct;
	GPIO_InitStruct.Pin   = LED_PIN;
	GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull  = GPIO_PULLUP;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(LED_GPIO_PORT, &GPIO_InitStruct);
}

int main(void)
{
	init();

	// HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN, GPIO_PIN_SET);
	// while (1) {}

	while (1)
	{
		// HAL_GPIO_TogglePin(LED_GPIO_PORT, LED_PIN);
		// HAL_Delay(1000);

		HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN, GPIO_PIN_RESET);
		HAL_Delay(200);
		HAL_GPIO_WritePin(LED_GPIO_PORT, LED_PIN, GPIO_PIN_SET);
		HAL_Delay(200);
	}
}
