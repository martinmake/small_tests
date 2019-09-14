#include <iostream>

template <typename T>
void binary_print(T value)
{
	std::cout << "0b";
	for (T mask = 1 << (sizeof(T) * 8 - 1); mask; mask >>= 1)
	{
		if (value & mask) std::cout << "1";
		else              std::cout << "0";
	}
	std::cout << std::endl;
}

int main(void)
{
	uint8_t  low_byte = 0b00000001;
	uint8_t high_byte = 0b00000001;

	uint16_t word = (high_byte << 8) | low_byte;

	binary_print( low_byte);
	binary_print(high_byte);
	binary_print(     word);
}
