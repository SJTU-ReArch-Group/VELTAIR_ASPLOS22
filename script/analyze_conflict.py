f = open("../build/lbl_conflict", "r").readlines()
import matplotlib.pyplot as plt 
downgrade = 0.0
time_dict = [[], [], [], [], [], []]
for _f in f:
    tmp = _f.split("\n")[0].split()
    time, demand, real = float(tmp[0]), float(tmp[1]), float(tmp[2])
    plt.plot([time, time], [real, demand], c='red')
    demandtime = time * real / demand
    downgrade += demandtime * (demand - real) / real
    time_idx = int(time / 200.0)
    time_dict[time_idx].append(demand-real)
print(downgrade / len(f))

dict_time = [[], [], [], [], [], []]
g = open("../build/sub_model_conflict", "r").readlines()
downgrade = 0.0
for _f in g:
    tmp = _f.split("\n")[0].split()
    time, demand, real = float(tmp[0]), float(tmp[1]), float(tmp[2])
    plt.plot([time, time], [real, demand], c='blue')
    demandtime = time * real / demand
    downgrade += demandtime * (demand - real) / real
    time_idx = int(time / 200.0)
    dict_time[time_idx].append(demand-real)
print(downgrade / len(g))
print(len(g), len(f))
for i in range(len(dict_time)):
    tmp_dict = {}
    for j in dict_time[i]:
        if j not in tmp_dict:
            tmp_dict[j] = 0
        tmp_dict[j] += 1
    print(i*200, (i + 1) * 200, tmp_dict)
plt.show()
