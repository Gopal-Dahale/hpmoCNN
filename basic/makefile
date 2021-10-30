SOURCE	= src/utils.cu src/user_iface.cu src/solver.cu src/neural_net.cu src/layer_params.cu src/main.cu
HEADER	= 
OUT	= main
CC	 = nvcc
INC_DIR = ./include
FLAGS	 = -I $(INC_DIR)
LFLAGS	 = 

all: $(SOURCE)
	$(CC) -o $(OUT) $(FLAGS) $(SOURCE) -lcublas -lcudnn -lcurand

clean: