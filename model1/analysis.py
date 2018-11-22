import numpy as np
from collections import Counter


def count_values():
    f = open("./data/data.txt", 'r')
    elements, strings = {}, []

    for idx, line in enumerate(f):
        if idx == 0:
            labels = line.split(',')
            continue
        strings.append(line.split(','))
    labels[-1] = 'atd'
    strings = np.transpose(strings)
    for idx, label in enumerate(labels):
        elements[label] = strings[idx]
    for idx, atd in enumerate(elements['atd']):
        elements['atd'][idx] = atd.replace('\n', '')
    for label in elements.keys():
        for idx, value in enumerate(elements[label]):
            elements[label] = Counter(elements[label])
    for label in elements.keys():
        print('[{}]'.format(label))
        for value in elements[label]:
            print('{}: {}'.format(value, elements[label][value]))
        print()


if __name__ == '__main__':
    count_values()
