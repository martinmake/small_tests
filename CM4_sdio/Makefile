# PROJECT DIRS
IDIRS:=inc
SDIR :=src
ODIR :=obj
DDIR :=dep
BDIR :=bin

# PROJECT FILES
ELF:=$(BDIR)/firmware.elf
HEX:=$(BDIR)/firmware.hex
BIN:=$(BDIR)/firmware.bin
ASM:=$(BDIR)/firmware.asm

# LIB DIRS
# LIB_DIR :=$(HOME)/Documents/proj/ARM/libs
# LIB_IDIR:=$(LIB_DIR)/inc
# LIB_ADIR:=$(LIB_DIR)/arc

# LIBS AND DEFS
LIBS:=
DEFS:=

# DRV DIRS
DRV_DIR  :=drivers
DRV_ARC  :=$(DRV_DIR)/lib$(DRV_DIR).a
DRV_IDIRS:=$(DRV_DIR)/CMSIS $(DRV_DIR)/STM32F4xx_StdPeriph_Driver/inc

# TARGET
LDSCRIPTS:=mem.ld layout.ld
TARGET   :=-mcpu=cortex-m4 -mthumb -mfloat-abi=softfp -fsingle-precision-constant
DEVICE   :=STM32F40XX

# TOOLCHAIN
TOOLCHAIN:=arm-none-eabi-
CC     :=$(TOOLCHAIN)gcc
CXX    :=$(TOOLCHAIN)g++
ASMC   :=$(TOOLCHAIN)gcc
LD     :=$(TOOLCHAIN)g++
OBJCOPY:=$(TOOLCHAIN)objcopy
OBJDUMP:=$(TOOLCHAIN)objdump
SIZE   :=$(TOOLCHAIN)size
STRIP  :=$(TOOLCHAIN)strip

# COMPILER FLAGS
WARNING     :=-Wall -Wextra -pedantic -Wno-write-strings
OPTIMIZATION:=-O3 -Os -fdata-sections -ffunction-sections -fno-exceptions

export TARGET DEVICE

######END#OF#CONFIGURATION#VARIABLES######

# LIBS+=standard
DEFS+=$(EDEFS)
DEFS+=$(DEVICE)
# DEFS+=F_CPU=$(F_CPU)
# DEFS+=BAUD=$(BAUD)

LIBS  :=$(addprefix -l,$(LIBS))
DEFS  :=$(addprefix -D,$(DEFS))

C_SRCS  :=$(shell find $(SDIR) -name '*.c'  )
CXX_SRCS:=$(shell find $(SDIR) -name '*.cpp')
ASM_SRCS:=$(shell find $(SDIR) -name '*.S'  )

C_OBJS  :=$(patsubst $(SDIR)/%.c,  $(ODIR)/%.o,$(C_SRCS)  )
CXX_OBJS:=$(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(CXX_SRCS))
ASM_OBJS:=$(patsubst $(SDIR)/%.S,  $(ODIR)/%.o,$(ASM_SRCS))
OBJS    :=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)

INC:=$(addprefix -I,$(IDIRS) $(LIB_IDIR) $(DRV_IDIRS))

LDSCRIPTS:=$(addprefix -T,$(LDSCRIPTS))

CCFLAGS  :=$(WARNING) $(OPTIMIZATION) -s $(TARGET) $(INC) $(DEFS) -std=c99
CXXFLAGS :=$(WARNING) $(OPTIMIZATION) -s $(TARGET) $(INC) $(DEFS) -std=c++14 -fno-rtti
ASMCFLAGS:=$(WARNING) $(OPTIMIZATION) -s $(TARGET) $(INC) $(DEFS)
LDFLAGS  :=$(WARNING) $(OPTIMIZATION)    $(TARGET) $(LDSCRIPTS) -Wl,--gc-sections --specs=nosys.specs

.SECONDARY: $(OBJS)

.PHONY: all
all: $(ELF)
	@echo "> BULDING"
	$(OBJCOPY) -O ihex   $(ELF)   $(HEX)
	$(OBJCOPY) -O binary $(ELF)   $(BIN)
	$(OBJDUMP) -Dz       $(ELF) > $(ASM)
	$(SIZE) $(HEX) | sed -e 's/^/* /'
	@echo "< BULDING DONE"

$(BIN): $(ELF)

$(ELF): $(OBJS) $(DRV_ARC)
	@echo "> LINKING"
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "< LINKING DONE"

$(ODIR)/%.o: $(SDIR)/%.c
	@echo "> COMPILING $<"
	$(CC)   $(CCFLAGS)   -c $< -o $@
	@echo "< COMPILING $< DONE"
$(ODIR)/%.o: $(SDIR)/%.cpp
	@echo "> COMPILING $<"
	$(CXX)  $(CXXFLAGS)  -c $< -o $@
	@echo "< COMPILING $< DONE"
$(ODIR)/%.o: $(SDIR)/%.S
	@echo "> COMPILING $<"
	$(ASMC) $(ASMCFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"
$(DRV_ARC):
	@echo "> COMPILING DRIVERS"
	+make -C $(DRV_DIR) | sed 's/^/\t/'
	@echo "< COMPILING DRIVERS DONE"

$(DDIR)/%.d: $(SDIR)/%.c
	@echo "> BUILDING DEPENDENCIES $<"
	$(CC)   $(CCFLAGS)   -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCIES $< DONE"
$(DDIR)/%.d: $(SDIR)/%.cpp
	@echo "> BUILDING DEPENDENCIES $<"
	$(CXX)  $(CXXFLAGS)  -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCIES $< DONE"
$(DDIR)/%.d: $(SDIR)/%.S
	@echo "> BUILDING DEPENDENCIES $<"
	$(ASMC) $(ASMCFLAGS) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCIES $< DONE"

.PHONY: flash
flash: $(BIN)
	@echo "> FLASHING $<"
	sudo st-flash write $< 0x08000000
	@echo "< FLASHING $< DONE"

.PHONY: serial
serial:
	@echo "> SERIAL MONITOR"
	sudo tio -b $(BAUD) -d 8 -s 2 -e -mINLCRNL $(PORT)
	@echo "> SERIAL MONITOR DONE"

.PHONY: setup
setup:
	@echo "> SETTING UP PROJECT"
	mkdir -p $(BDIR) $(ODIR) $(DDIR) $(IDIRS) $(SDIR)
	touch $(BDIR)/.empty $(ODIR)/.empty $(DDIR)/.empty
	@echo "< SETTING UP PROJECT DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vfr ./$(ODIR)/* ./$(DDIR)/* $(ELF) $(HEX) $(BIN) $(ASM) | sed -e 's/^/* /'
	@echo "< CLEANING DONE"
	@echo "> CLEANING DRIVERS"
	+make -C $(DRV_DIR) clean | sed 's/^/\t/'
	@echo "< CLEANING DRIVERS DONE"

ifeq ($(NODEP),)
-include $(patsubst $(SDIR)/%.c,  $(DDIR)/%.d,$(C_SRCS)  )
-include $(patsubst $(SDIR)/%.cpp,$(DDIR)/%.d,$(CXX_SRCS))
-include $(patsubst $(SDIR)/%.S,  $(DDIR)/%.d,$(ASM_SRCS))
endif
