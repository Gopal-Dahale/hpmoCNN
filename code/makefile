NN_DIR = src/neural_net
DATASETS_DIR = src/datasets
NEURAL_NET = $(NN_DIR)/neural_net.cu $(NN_DIR)/get_loss.cu $(NN_DIR)/compute_loss.cu $(NN_DIR)/compare_output_correct.cu $(NN_DIR)/save_and_load.cu $(NN_DIR)/forward_prop.cu $(NN_DIR)/backward_prop.cu
DATASETS = $(DATASETS_DIR)/mnist224.cu 
SOURCE	= src/utils.cu src/user_iface.cu src/solver.cu $(NEURAL_NET) $(DATASETS) src/layer_params.cu src/main.cu src/metrics.cu src/parser.cu
HEADER	= 
OUT	= main
CC	 = nvcc
INC_DIR_BASIC = ./include/
INC_DIR_EXTERN = ../cxxopts/include/ 
INC_DIR_LOGGER = ../plog/include/
FLAGS	 = -I$(INC_DIR_BASIC) -I$(INC_DIR_EXTERN) -I$(INC_DIR_LOGGER)
LOG_FILE_NAME = Log.txt
LFLAGS	 = 

all: $(SOURCE)
	rm -r $(LOG_FILE_NAME)
	$(CC) -o $(OUT) $(FLAGS) $(SOURCE) -lcublas -lcudnn -lcurand

clean: