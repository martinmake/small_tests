#include <stdint.h>
#include <system_stm32f10x.h>
#include <stm32f10x_rcc.h>
#include <stm32f10x_gpio.h>

#define LED1        GPIO_Pin_8
#define LED1PORT    GPIOA
#define LED1PORTCLK RCC_APB2Periph_GPIOA
#define LED2        GPIO_Pin_2
#define LED2PORT    GPIOD
#define LED2PORTCLK RCC_APB2Periph_GPIOD

void delay(uint64_t n)
{
	while (n--)
		for (uint16_t i = 10000; i; i--)
			asm("NOP");
}

void init(void)
{
	RCC_APB2PeriphClockCmd(LED1PORTCLK, ENABLE);
	RCC_APB2PeriphClockCmd(LED2PORTCLK, ENABLE);

	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;

	GPIO_InitStructure.GPIO_Pin = LED1;
	GPIO_Init(LED1PORT, &GPIO_InitStructure);
	GPIO_InitStructure.GPIO_Pin = LED2;
	GPIO_Init(LED2PORT, &GPIO_InitStructure);
}

int main(void)
{
	init();

	while (1) {
	   	GPIO_ResetBits(LED1PORT, LED1);
	   	GPIO_SetBits(  LED2PORT, LED2);
	   	delay(100);
	   	GPIO_SetBits(  LED1PORT, LED1);
	   	GPIO_ResetBits(LED2PORT, LED2);
	   	delay(100);
        }
}
