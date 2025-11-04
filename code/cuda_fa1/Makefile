NVCC  := nvcc
ARCH  := sm_80
FLAGS := -O2 -arch=$(ARCH)
TARGET := flashAttention
MAIN_TARGET := main
SRC    := flashAttention.cu
MAIN_SRC := main.cu

B     := 1
H     := 1
N     := 4096
D     := 64
M     := 16384
RUNS  := 1

.DEFAULT_GOAL := all

$(MAIN_TARGET): $(MAIN_SRC) $(SRC)
	$(NVCC) $(FLAGS) $(MAIN_SRC) $(SRC) -o $@

all: $(MAIN_TARGET)
	@./$(MAIN_TARGET)

run_main: $(MAIN_TARGET)
	@./$(MAIN_TARGET) $(B) $(H) $(N) $(D) $(M) $(RUNS)

clean:
	rm -f $(TARGET) $(MAIN_TARGET)
