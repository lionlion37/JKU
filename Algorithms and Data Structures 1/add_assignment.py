import os
import argparse


def add_assignment():
    parser = argparse.ArgumentParser(description='add assignment')
    parser.add_argument('ass_nr', type=int, help='Number of assignment to add')
    args = parser.parse_args()
    ass_nr = str(args.ass_nr)

    if not os.path.isdir(os.path.join('.', 'assignments', 'k01553060-assignment0' + ass_nr)):
        os.system(f'mkdir {os.path.join(".", "assignments", "k01553060-assignment0" + ass_nr)}')
        print('Successfully created assignment folder!')
    else:
        print('Assignment folder already exists.')

    if not os.path.isfile(os.path.join('.', 'time_logs', 'time_' + ass_nr + '.txt')):
        with open(os.path.join('.', 'time_logs', 'time_' + ass_nr + '.txt'), 'w') as f:
            f.write('#Student ID\n'
                    'k01553060\n'
                    '#Assignment number\n'
                    f'0{ass_nr}\n'
                    '#Time in minutes\n')
        print('Successfully created time file!')
    else:
        print('Time file already exists.')


if __name__ == '__main__':
    add_assignment()