"""Runnable Sobel version; sweep pilot weights by default."""
from pilot import main

if __name__ == "__main__":
    main(default_weights=[1000, 10000, 100000])
