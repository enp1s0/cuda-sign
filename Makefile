NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I./cutf
TARGET=sign_test

$(TARGET):main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

ptx: main.cu
	$(NVCC) $(NVCCFLAGS) $< -ptx


clean:
	rm -f $(TARGET)
