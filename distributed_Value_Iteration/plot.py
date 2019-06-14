import numpy as np
import matplotlib.pyplot as plt

map_size = [8, 16, 32]
time_single_v1 = [0.844080924987793, 15.268503189086914, 407.7942831516266]
time_single_v2 = [0.396909236907959, 1.7127573490142822, 13.425556659698486]
time_distributed_v1 = [13.29153823852539, 76.31096506118774, 945.1249794960022]
time_distributed_v2 = [1.2731595039367676, 1.9951863288879395, 6.614015817642212]

num_workers = [2, 4, 8] #map size 16
time_workers_distributed_v1 = [148.77010846138, 76.31096506118774, 56.38988137245178]
time_workers_distributed_v2 = [1.5744061470031738, 1.9951863288879395, 2.34501576423645]

#times for changing num_batches and num_workers
#map size 32x32

# nw=2
# b = [2,4,8,16,32]
# min_time = 10.598074197769165 at b=2
#
# nw=4
# b = [4,8,16,32]
# min_time = 8.731677055358887 at b=2
#
# nw=8
# b = [8,16,32]
# min_time = 8.902066469192505 at b=2
#
# nw=16
# b = [16,32]
# min_time =


fig = plt.figure()
plt.plot(map_size, time_single_v2, label="undistributed_v2", color='blue', marker='o')
plt.plot(map_size, time_distributed_v2, label="distributed_v2", color='red', marker='o')
# plt.plot(num_workers, time_workers_distributed_v1, label="distributed_v1", color='green', marker='o')
# plt.plot(num_workers, time_workers_distributed_v2, label="distributed_v2", color='red', marker='o')
plt.xlabel("Mapsize")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Mapsizes")
plt.legend(loc=1)
plt.show()
