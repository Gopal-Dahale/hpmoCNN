mkdir ../vgg116
mkdir ../vgg116/batch_size32
mkdir ../vgg116/batch_size64
mkdir ../vgg116/batch_size128
mkdir ../vgg116/batch_size256
cd ../vgg116/batch_size32
touch configs.txt
touch loss.txt
touch val_acc.txt
touch batch_times.txt
touch totaltime.txt
touch totaloverhead.txt
touch mem_usage.txt
touch offload_mem.txt
cd -
cd ../vgg116/batch_size64
touch configs.txt
touch loss.txt
touch val_acc.txt
touch batch_times.txt
touch totaltime.txt
touch totaloverhead.txt
touch mem_usage.txt
touch offload_mem.txt
cd -
cd ../vgg116/batch_size128
touch configs.txt
touch loss.txt
touch val_acc.txt
touch batch_times.txt
touch totaltime.txt
touch totaloverhead.txt
touch mem_usage.txt
touch offload_mem.txt
cd -
cd ../vgg116/batch_size256
touch configs.txt
touch loss.txt
touch val_acc.txt
touch batch_times.txt
touch totaltime.txt
touch totaloverhead.txt
touch mem_usage.txt
touch offload_mem.txt
cd -
make -f makefile.vgg116
./main --batch-size 64 --epochs 10 --nn 116
echo "batch size 64 done"
./main --batch-size 128 --epochs 10 --nn 116
echo "batch size 128 done"
./main --batch-size 256 --epochs 10 --nn 116
echo "batch size 256 done"