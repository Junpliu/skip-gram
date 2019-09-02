import os
def organize_semantic_eval():
    root = 'evaluation/Semantic_questions'
    file_path = [root + '/training', root + '/testing']
    with open('semevalQuestion1.txt', 'w') as ques:
        with open('semevalAnswer1.txt', 'w') as answ:
            for path in file_path:
                for file in os.listdir(path):
                    print(file)
                    with open(path + '/' + file, 'r') as g:
                        lines = g.read().split('\n')
                        flag = 0
                        base_pair = []
                        com_pair = []
                        for line in lines:
                            line_items = line.split()
                            if len(line_items) != 0 and line_items[0] != '#' and float(line_items[0]) >= 20:
                                if flag <= 1:
                                    base_pair.append(line_items[1][1:-1].split(':'))
                                    flag += 1
                                com_pair.append(line_items[1][1:-1].split(':'))
                                com_pair[-1].append(line_items[0])
                        print('base: ', base_pair)
                        print('com pair: ', com_pair)
                        for base_idx in range(len(base_pair)):
                            for com_idx in range(len(com_pair)):
                                ques.write(' '.join(base_pair[base_idx]) + ' ' + com_pair[com_idx][0] + '\n')
                                answ.write(com_pair[com_idx][-1] + ' ' + com_pair[com_idx][1] + '\n')
                                ques.write(' '.join(base_pair[base_idx][::-1]) + ' ' + com_pair[com_idx][1] + '\n')
                                answ.write(com_pair[com_idx][-1] + ' ' + com_pair[com_idx][0] + '\n')


if __name__ == '__main__':
    organize_semantic_eval()