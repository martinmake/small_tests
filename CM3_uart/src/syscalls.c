#include  <errno.h>
#include  <sys/unistd.h>
#include  <stm32f10x_usart.h>

int _write(int file, char *data, int len)
{
	int bytes_written;

	if ((file != STDOUT_FILENO) && (file != STDERR_FILENO))
	{
		errno = EBADF;
		return -1;
	}

	for (bytes_written = 0; bytes_written < len; bytes_written++)
	{
		 USART_SendData(USART1, *data);
		 data++;
	}

	return bytes_written;
}
