"""This is the main entry point for the gui and the processing pipeline working with
real time data. The nuc_code.py file has to run on a system connected to the sensors."""
from production.app import main

if __name__ == "__main__":
    main()