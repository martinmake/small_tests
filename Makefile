LDIR :=$(HOME)/Documents/libs/AVR
LADIR:=$(LDIR)/arc
LIDIR:=$(LDIR)/inc
LMDIR:=$(LDIR)/man

######END#OF#CONFIGURATION#VARIABLES######

export LDIR LADIR LIDIR LMDIR

PATHS:=$(dir $(shell find ./*/* -name Makefile))
EXCLUDE_PATHS:=$(dir $(shell find ./*/* -name Makefile | grep drivers))
PATHS:=$(filter-out $(EXCLUDE_PATHS),$(PATHS))

.PHONY: all
all:
	+$(foreach PATH,$(PATHS),make -e -C $(PATH) -j4;)


.PHONY: clean
clean:
	+$(foreach PATH,$(PATHS),make -e -C $(PATH) -j4 clean;)
