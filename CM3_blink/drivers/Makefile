# PROJECT DIRS
IDIRS:=CMSIS STM32F10x_StdPeriph_Driver/inc

# PROJECT FILES
ARC:=lib$(notdir $(realpath .)).a

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
WARNING     :=-Wall -Wextra -pedantic -Wno-implicit-function-declaration -Wno-unused-parameter
OPTIMIZATION:=-fdata-sections -ffunction-sections -fno-exceptions

######END#OF#CONFIGURATION#VARIABLES######

ifeq ($(TARGET)$(DEVICE),)
$(error DO NOT RUN THIS MAKEFILE DIRECTLY!)
endif

DEFS+=$(DEVICE)
DEFS:=$(addprefix -D,$(DEFS))

C_SRCS  :=$(shell find -name '*.c'  )
CXX_SRCS:=$(shell find -name '*.cpp')
ASM_SRCS:=$(shell find -name '*.S'  )

C_OBJS  :=$(patsubst %.c,  %.o,$(C_SRCS)  )
CXX_OBJS:=$(patsubst %.cpp,%.o,$(CXX_SRCS))
ASM_OBJS:=$(patsubst %.S,  %.o,$(ASM_SRCS))
OBJS    :=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)

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
	$(CC)   $(CCFLAGS)  -c $< -o $@
	@echo "< COMPILING $< DONE"
%.o:%.cpp
	@echo "> COMPILING $<"
	$(CXX)  $(CXXFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"
%.o:%.S
	@echo "> COMPILING $<"
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vf $(OBJS) $(ARC) | sed -e 's/^/* /'
	@echo "< CLEANING DONE"
