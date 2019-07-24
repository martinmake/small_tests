# PROJECT DIRS
IDIRS:=. CMSIS STM32F1xx_HAL_Driver/Inc

# PROJECT FILES
ARC:=lib$(notdir $(realpath .)).a

# TARGET
TARGET:=-mcpu=cortex-m3 -mthumb -mfloat-abi=softfp -fsingle-precision-constant

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
SRCS:=$(shell find -name '*.[cS]' | grep -v '^./_')
OBJS:=$(SRCS)
OBJS:=$(OBJS:.c=.o)
OBJS:=$(OBJS:.S=.o)
INC :=$(addprefix -I,$(IDIRS))

CCFLAGS  :=$(WARNING) $(OPTIMIZATION) $(TARGET) $(INC) $(DEFS) -std=c99
ASMCFLAGS:=$(WARNING) $(OPTIMIZATION) $(TARGET) $(INC) $(DEFS)
ARFLAGS  :=rUucv

.SECONDARY: $(OBJS)

.PHONY: all
all: $(ARC)

$(ARC): $(OBJS)
	@echo "> ARCHIVING"
	$(AR) $(ARFLAGS) $@ $^
	@echo "< ARCHIVING DONE"

%.o:%.c
	@echo "> COMPILING $<"
	$(CC)   $(CCFLAGS)   -c $< -o $@
	@echo "< COMPILING $< DONE"
%.o:%.S
	@echo "> COMPILING $<"
	$(ASMC) $(ASMCFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vf $(OBJS) $(ARC) | sed -e 's/^/* /'
	@echo "< CLEANING DONE"
