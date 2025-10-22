echo 'TP4PP2_warmup' | tee result.log
date >> result.log
python3 TP4PP2_warmup.py
echo 'Warmup completed' | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP4PP2_latency' | tee result.log
date >> result.log
python3 TP4PP2_latency.py | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP4PP2_throughput' | tee -a result.log
date >> result.log
python3 TP4PP2_throughput.py | tee -a result.log
date >> result.log