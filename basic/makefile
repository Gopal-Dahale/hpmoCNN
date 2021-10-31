NN_DIR = src/neural_net
NEURAL_NET = $(NN_DIR)/neural_net.cu $(NN_DIR)/get_loss.cu $(NN_DIR)/compute_loss.cu $(NN_DIR)/compare_output_correct.cu
SOURCE	= src/utils.cu src/user_iface.cu src/solver.cu $(NEURAL_NET) src/layer_params.cu src/main.cu
HEADER	= 
OUT	= main
CC	 = nvcc
INC_DIR = ./include
FLAGS	 = -I $(INC_DIR)
LFLAGS	 = 

all: $(SOURCE)
	$(CC) -o $(OUT) $(FLAGS) $(SOURCE) -lcublas -lcudnn -lcurand

clean: