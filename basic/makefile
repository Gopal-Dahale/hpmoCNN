SOURCE	= src/main.cu src/user_iface.cu 
HEADER	= 
OUT	= main
CC	 = nvcc
INC_DIR = ./include
FLAGS	 = -I $(INC_DIR)
LFLAGS	 = 

all: $(SOURCE)
	$(CC) -o $(OUT) $(FLAGS) $(SOURCE) 

clean: