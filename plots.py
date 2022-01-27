import matplotlib.pyplot as plt

with open("results.txt", 'r') as file:
    linesraw = file.readlines()

lines = []
for line in linesraw:
    words = line.split()
    lines.append(words)

yCthreads = []
yOMP = []
temp = ''
for element in lines:
    if element[0] == 'CTHREADS' or element[0] == 'OMP':
        temp = element[0]
        continue
    else:
        if temp == 'CTHREADS':
            yCthreads.append(float(element[0]))
        else:
            yOMP.append(float(element[0]))

x = range(2,17,2)
yNr= range(0,10)
plt.plot(x,yCthreads, '--*', label = 'C++ Threads')
plt.plot(x, yOMP, '--*', label = 'OMP')
plt.axhline(y=1, color='black', linestyle='-', label='Baseline')

plt.title('Performance comparison of C++ Threads and OMP')
plt.ylabel('Speedup compared to Baseline')
plt.xlabel('# of threads')
plt.yticks(yNr)
plt.legend()
plt.savefig("performance.jpg")
