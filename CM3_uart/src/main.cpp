#include <stdint.h>
#include <stdio.h>
#include <system_stm32f10x.h>
#include <stm32f10x_rcc.h>
#include <stm32f10x_gpio.h>
#include <stm32f10x_usart.h>

#define LED           GPIO_Pin_8
#define LED_GPIO      GPIOA
#define LED_GPIO_CLK  RCC_APB2Periph_GPIOA

#define USART           USART1
#define USART_TX        GPIO_Pin_9
#define USART_RX        GPIO_Pin_10
#define USART_IT_RXEN   USART1_IT_RXEN
#define USART_IRQn      USART1_IRQn
#define USART_CLK       RCC_APB2Periph_USART1
#define USART_GPIO      GPIOA
#define USART_GPIO_CLK  RCC_APB2Periph_GPIOA

void init_led(void)
{
	RCC_APB2PeriphClockCmd(LED_GPIO_CLK, ENABLE);

	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Pin = LED;
	GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(LED_GPIO, &GPIO_InitStructure);

	GPIO_SetBits(LED_GPIO, LED);
}

void init_usart(void)
{
	RCC_APB2PeriphClockCmd(USART_GPIO_CLK | USART_CLK | RCC_APB2Periph_AFIO,  ENABLE);

	GPIO_InitTypeDef  GPIO_InitStructure;
	USART_InitTypeDef USART_InitStructure;

	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	// Configure USART Tx
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_InitStructure.GPIO_Pin  = USART_TX;
	GPIO_Init(USART_GPIO, &GPIO_InitStructure);
	// Configure USART Rx
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
	GPIO_InitStructure.GPIO_Pin  = USART_RX;
	GPIO_Init(USART_GPIO, &GPIO_InitStructure);

	USART_Cmd(USART, ENABLE);

	USART_InitStructure.USART_BaudRate            = TIO_BAUD;
	USART_InitStructure.USART_WordLength          = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits            = USART_StopBits_1;
	USART_InitStructure.USART_Parity              = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode                = USART_Mode_Rx | USART_Mode_Tx;
	USART_Init(USART, &USART_InitStructure);

	USART_ITConfig(USART, USART_IT_RXNE, ENABLE);
	NVIC_EnableIRQ(USART_IRQn);
}

void init(void)
{
	init_usart();
	init_led();
}

void led_toggle(void)
{
	if (GPIO_ReadOutputDataBit(LED_GPIO, LED) == Bit_SET)
		GPIO_ResetBits(LED_GPIO, LED);
	else
		GPIO_SetBits(  LED_GPIO, LED);
}

int main(void)
{
	init();

	// RCC_ClocksTypeDef RCC_Clocks;

	led_toggle();
	while (1) {
		led_toggle();
		USART1->DR = '*';
		for (uint64_t i = 1000000; i; i--)
			__asm volatile ("NOP");

	//	for (char* str = "TEST\n"; *str; str++) {
	//		while (USART_GetFlagStatus(USART, USART_FLAG_TXE) == RESET) {}
	//		USART_SendData(USART, *str);
	//	}
	//	RCC_GetClocksFreq(&RCC_Clocks);
	//	printf("Running at %luMHz\n", RCC_Clocks.SYSCLK_Frequency / 1000000);
        }
}

void USART1_IRQHandler(void)
{
	/* RXNE handler */
	if (USART_GetITStatus(USART, USART_IT_RXNE) != RESET)
		led_toggle();

	//USART_ClearITStatus(USART, USART_IT_RXNE);
}
