# PROJECT DIRS
IDIRS:=CMSIS STM32F10x_StdPeriph_Driver/inc

# PROJECT FILES
ARC:=lib$(notdir $(realpath .)).a

# TARGET
TARGET:=-mcpu=cortex-m3 -mthumb -mfloat-abi=softfp -fsingle-precision-constant
DEVICE:=STM32F10X_HD

# TOOLCHAIN
TOOLCHAIN:=arm-none-eabi-
CC     :=$(TOOLCHAIN)gcc
CXX    :=$(TOOLCHAIN)g++
ASMC   :=$(TOOLCHAIN)gcc
LD     :=$(TOOLCHAIN)g++
AR     :=$(TOOLCHAIN)ar
OBJCOPY:=$(TOOLCHAIN)objcopy
OBJDUMP:=$(TOOLCHAIN)objdump
SIZE   :=$(TOOLCHAIN)size
STRIP  :=$(TOOLCHAIN)strip

# COMPILER FLAGS
WARNING     :=-Wall -Wextra -pedantic -Wno-implicit-function-declaration
OPTIMIZATION:=-fdata-sections -ffunction-sections -fno-exceptions

######END#OF#CONFIGURATION#VARIABLES######

DEFS+=$(DEVICE)
DEFS:=$(addprefix -D,$(DEFS))
SRCS:=$(shell find -name '*.c')
OBJS:=$(SRCS:.c=.o)
INC :=$(addprefix -I,$(IDIRS))

CCFLAGS:=$(WARNING) $(OPTIMIZATION) $(TARGET) $(INC) $(DEFS) -std=c99
ARFLAGS:=rUucv

.SECONDARY: $(OBJS)

.PHONY: all
all: $(ARC)

$(ARC): $(OBJS)
	@echo "> ARCHIVING"
	$(AR) $(ARFLAGS) $@ $^
	@echo "< ARCHIVING DONE"

%.o:%.c
	@echo "> COMPILING $<"
	$(CC) $(CCFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vf $(OBJS) $(ARC) | sed -e 's/^/* /'
	@echo "< CLEANING DONE"
