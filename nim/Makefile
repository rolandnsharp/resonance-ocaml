.PHONY: build clean train

build: src/kernels.o
	nim c -d:release --path:src src/train.nim

src/kernels.o: src/kernels.cu
	nvcc -c -O2 -Xcompiler -fPIC -o src/kernels.o src/kernels.cu

train: build
	./src/train ../data/shakespeare.txt 96 6 3000

clean:
	rm -f src/kernels.o src/train
