for((i=0;i<20;i++))
do
	wget csie.ntu.edu.tw/~b04902004/models/model${i}.hdf5
done
python3 best.py $1 $2
