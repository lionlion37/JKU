import argparse
import sys
import select
import os
import time


def log_time():
    parser = argparse.ArgumentParser(description='Log time for assignment')
    parser.add_argument('ass_nr', type=int)
    args = parser.parse_args()
    ass_nr = str(args.ass_nr)

    if not os.path.isfile(os.path.join('.', 'time_logs', 'time_' + ass_nr + '.txt')):
        raise FileNotFoundError('No existing time log file found. Please create one by using add_assignment.')

    print('Started timer!\n\npause:   "p"\nrestart: "r"\nstop:    "s"\n')
    start_time = time.time()
    finished = False
    while not finished:
        c_t = time.time() - start_time
        if c_t < 60:
            sys.stdout.write('\r')
            sys.stdout.write(f'{int(round(c_t, 0))} s ')
            sys.stdout.flush()
        else:
            sys.stdout.write('\r')
            sys.stdout.write(f'{int(c_t // 60)} m {int(round(c_t % 60, 0))} s ')
            sys.stdout.flush()

        if select.select([sys.stdin,],[],[],0.0)[0]:
            instruction = sys.stdin.readline()[0]

            if instruction == 'p':
                sys.stdout.write('\r')
                sys.stdout.write('Paused! To return type "r"\n')
                sys.stdout.flush()
                p_t_start = time.time()
                con = False
                while not con:
                    instruction = input()
                    if instruction == 'r':
                        sys.stdout.write('\r')
                        sys.stdout.write('Returned!\n')
                        sys.stdout.flush()
                        p_t = time.time() - p_t_start
                        start_time = start_time + p_t
                        con = True

            if instruction == 's':
                with open(os.path.join('.', 'time_logs', 'time_' + ass_nr + '.txt'), 'a') as f:
                    if c_t > 60:
                        f.write(str(int(c_t // 60)))
                        sys.stdout.write('\r')
                        sys.stdout.write(f'\nTime in minutes: {int(c_t // 60)}\n\n')
                        sys.stdout.flush()
                    else:
                        sys.stdout.write('\r')
                        sys.stdout.write('Recorded time too short to log (must be at least 1 min)!\n')
                        sys.stdout.flush()
                finished = True
                break

        time.sleep(1)


if __name__ == '__main__':
    log_time()