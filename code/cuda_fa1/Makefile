NVCC  := nvcc
ARCH  := sm_80
FLAGS := -O2 -arch=$(ARCH)
TARGET := flashAttention
MAIN_TARGET := main
SRC    := flashAttention.cu
MAIN_SRC := main.cu

B     := 1
H     := 32
N     := 8192
D     := 128
M     := 16384
RUNS  := 10

all: $(TARGET) $(MAIN_TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(FLAGS) $< -o $@

$(MAIN_TARGET): $(MAIN_SRC) $(SRC)
	$(NVCC) $(FLAGS) $(MAIN_SRC) $(SRC) -o $@

run: $(TARGET)
	@./$(TARGET) $(B) $(H) $(N) $(D) $(M) $(RUNS)

run_main: $(MAIN_TARGET)
	@./$(MAIN_TARGET) $(B) $(H) $(N) $(D) $(M) $(RUNS)

clean:
	rm -f $(TARGET) $(MAIN_TARGET)