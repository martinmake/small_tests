include Makefile.config

DEFS+=$(DEVICE)
DEFS:=$(addprefix -D,$(DEFS))
SRCS:=$(shell find -name '*.c' | grep -v '^./_') $(STARTUP)
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
