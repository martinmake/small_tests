#include <stm32f1xx.h>

#define LED1_GPIO_CLK_ENABLE() __HAL_RCC_GPIOA_CLK_ENABLE();
#define LED1_GPIO_PORT         GPIOA
#define LED1_PIN               GPIO_PIN_8
#define LED2_GPIO_CLK_ENABLE() __HAL_RCC_GPIOD_CLK_ENABLE();
#define LED2_GPIO_PORT         GPIOD
#define LED2_PIN               GPIO_PIN_2

void GPIO_Init(void)
{
	LED1_GPIO_CLK_ENABLE();
	LED2_GPIO_CLK_ENABLE();

	GPIO_InitTypeDef GPIO_InitStruct;
	GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull  = GPIO_PULLUP;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

	GPIO_InitStruct.Pin = LED1_PIN;
	HAL_GPIO_Init(LED1_GPIO_PORT, &GPIO_InitStruct);

	GPIO_InitStruct.Pin = LED2_PIN;
	HAL_GPIO_Init(LED2_GPIO_PORT, &GPIO_InitStruct);

	HAL_GPIO_WritePin(LED1_GPIO_PORT, LED1_PIN, GPIO_PIN_SET);
	HAL_GPIO_WritePin(LED2_GPIO_PORT, LED2_PIN, GPIO_PIN_SET);
}

int main(void)
{
	HAL_Init();
	GPIO_Init();

	HAL_GPIO_WritePin(LED2_GPIO_PORT, LED2_PIN, GPIO_PIN_RESET);
	while (1)
	{
		HAL_GPIO_TogglePin(LED1_GPIO_PORT, LED1_PIN);
		HAL_GPIO_TogglePin(LED2_GPIO_PORT, LED2_PIN);
		HAL_Delay(100);
	}
}
