echo 'TP1_warmup' | tee result.log
date >> result.log
python3 TP1_warmup.py
echo 'Warmup completed' | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP1_latency' | tee result.log
date >> result.log
python3 TP1_latency.py | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP1_throughput' | tee -a result.log
date >> result.log
python3 TP1_throughput.py | tee -a result.log
date >> result.log