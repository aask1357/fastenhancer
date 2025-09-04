import sys
# from array import array
# from fcntl import ioctl
# from termios import TIOCGWINSZ


def clear_previous_line():
    sys.stdout.write("\033[F")  # move cursor to the previous line
    sys.stdout.write("\033[K")  # clear line from the cursor
    print("\r", end="")         # move the cursor to the beginning of the line


def clear_current_line():
    # cols = array('h', ioctl(sys.stdout.fileno(), TIOCGWINSZ, '\0' * 8))[1]
    # print("\r" + " " * cols, flush=True, end="")
    sys.stdout.write("\033[2K") # clear the whole line
    print("\r", end="")         # move the cursor to the beginning of the line


if __name__=="__main__":
    print("000000000000000000000000000000000000")
    print("1first: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    print(f"\r1second: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    print(f"\r1third", flush=True)
    
    print("1111111111111111111111111111111111")
    
    print("2first: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    print(f"\r2second: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="\n")
    clear_previous_line()
    print(f"\r2third")
    
    print("22222222222222222222222222222222222222")
    
    print("3first: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    print(f"\r3second: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    clear_current_line()
    print(f"\r3third")
    
    print("3first: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    print(f"\r3second: afdfdsfasdfadsf  asfdasfasdfadsfasdf", end="")
    clear_current_line()
    print(f"3third")