OPENMP = 1
LIBSO = 0
AVX = 1

OS := $(shell uname)

EXEC = otter
OBJDIR = ./obj/
SRCDIR = ./Tensor/

ifeq ($(LIBSO), 1)
LIBNAMESO = otter.so
endif

CC = g++ -std=c++17
OPTS = -O2 -fno-finite-math-only
#CFLAGS = -Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC
CFLAGS = -fPIC

CFLAGS += $(OPTS)
LDFLAGS = -lm

ifeq ($(AVX), 1)
	CFLAGS += -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a -mfma
endif

ifeq ($(OPENMP), 1)
	ifeq ($(OS), Darwin)
		CFLAGS += -Xpreprocessor -fopenmp
		LDFLAGS += -lomp
	else
		CFLAGS += -fopenmp
	endif
endif

all: $(OBJDIR) backup $(EXEC) $(LIBNAMESO)

ifeq ($(LIBSO), 1)
CFLAGS+= -fPIC

$(LIBNAMESO): $(OBJDIR) $(OBJS)
	$(CC) -shared -std=c++14 -fvisibility=hidden $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

endif

SOURCES = $(wildcard $(SRCDIR)*.cpp)
OBJS = $(patsubst $(SRCDIR)%.cpp, $(OBJDIR)%.o, $(SOURCES))
DEPS = $(wildcard $(SRCDIR)*.hpp) $(SRCDIR)3rdparty/*.h

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: $(SRCDIR)%.cpp $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)
backup:
	mkdir -p backup

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)
