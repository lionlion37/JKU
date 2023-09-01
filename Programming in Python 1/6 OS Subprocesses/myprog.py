import argparse
parser = argparse.ArgumentParser()
parser.add_argument('myarg', type=int)

# Parse the arguments
args = parser.parse_args()

print(args.myarg)
