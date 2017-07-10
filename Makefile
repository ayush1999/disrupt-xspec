include $(PWD)/Makefile.in

INC = -I$(XSDIR)/include
XSLIB = -L$(XSDIR)/lib -lXSFunctions -lXSModel -lXSUtil -lXS
LIB = $(XSLIB) -lm

.PHONY: clean

SRC = test-interface.c

default: test-interface

test-interface.o: $(SRC)
	$(CC) $(INC) -o $@ -c $^

test-interface: test-interface.o
	$(CC) -o $@ $^ $(LIB)


clean:
	rm -f test-interface.o
	rm -f test-interface
