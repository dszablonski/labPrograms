#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:54:54 2024

@author: dominik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *

# File reading details
FILE_NAME = 'e3data.csv'  # Remember to include the extension
SKIP_FIRST_LINE = False
DELIMITER = ','  # Set to space, ' ', if working with .txt file without commas
Y_LABEL = r"V$_{\mathrm{out}} / V$"
X_LABEL = r"V$_{\mathrm{in}} / V$"
TITLE = ""#"Output Voltage vs. Input Voltage"
PLOT_TITLE= "plot3.pdf"

def function(x, a, b):
    return a*x + b

def new_f(B, x):
    return B[0]*x + B[1]

def check_numeric(entry):
    """Checks if entry is numeric
    Args:
        entry: string
    Returns:
        bool
    Raises:
        ValueError: if entry cannot be cast to float type
    """
    try:
        float(entry)
        return True
    except ValueError:
        return False

def check_uncertainty(uncertainty):
    """Checks uncertainty is non-zero and positive.
    Args:
        uncertainty: float
    Returns:
        Bool
    """
    if uncertainty > 0:
        return True
    return False


def validate_line(line):
    """Validates line. Outputs error messages accordingly.
    Args:
        line: string
    Returns:
        bool, if validation has been succesful
        line_floats, numpy array of floats
    """
    line_split = line.split(DELIMITER)

    for entry in line_split:
        if check_numeric(entry) is False:
            print('Line omitted: {0:s}.'.format(line.strip('\n')))
            print('{0:s} is nonnumerical.'.format(entry))
            return False, line_split
    line_floats = np.array([float(line_split[0]), float(line_split[1]), float(line_split[2]), float(line_split[3])])

    return True, line_floats


def open_file(file_name=FILE_NAME, skip_first_line=SKIP_FIRST_LINE):
    """Opens file, reads data and outputs data in numpy arrays.
    Args:
        file_name: string, default given by FILE_NAME
    Returns:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
    Raises:
        FileNotFoundError
    """
    # Create empty arrays ready to store the data
    x_data = np.array([])
    y_data = np.array([])
    x_err = np.array([])
    y_err = np.array([])
    try:
        raw_file_data = open(file_name, 'r')
    except FileNotFoundError:
        print("File '{0:s}' cannot be found.".format(file_name))
        print('Check it is in the correct directory.')
        return x_data, y_data
    for line in raw_file_data:
        if skip_first_line:
            skip_first_line = False
        else:
            line_valid, line_data = validate_line(line)
            if line_valid:
                x_data = np.append(x_data, line_data[0])
                y_data = np.append(y_data, line_data[1])
                x_err = np.append(x_err, line_data[2])
                y_err = np.append(y_err, line_data[3])
    raw_file_data.close()
    return x_data, y_data, x_err, y_err

def chi_squared_function(x_data, y_data, y_uncertainties, parameters):
    """Calculates the chi squared for the data given, assuming a linear
    relationship.
    Args:
        x_data: numpy array of floats
        y_data: numpy array of floats
        y_uncertainties: numpy array of floats
        parameters: numpy array of floats, [slope, offset]
    Returns:
        chi_squared: float
    """
    return np.sum((new_f(parameters, x_data)
                   - y_data)**2 / y_uncertainties**2)

def plotter(xdata, ydata, xfit, yfit, err_x, err_y):
    plt.plot(xfit, yfit, 'r-', label="Linear Fit" )    
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.errorbar(xdata, ydata, yerr=err_y, xerr=err_x, fmt='.', label="Data")
    plt.legend()
    #plt.title(TITLE)
    plt.savefig(fname=PLOT_TITLE, dpi=800)
    plt.show()

def main():
    x_data, y_data, x_err, y_err = open_file()
    
    linear = Model(new_f)
    mydata = RealData(x_data, y_data, sx=x_err, sy=y_err)
    myodr = ODR(mydata, linear, beta0=[0.65052890998032,0])
    myoutput = myodr.run()
    myoutput.pprint()
    
    x_fit = np.linspace(x_data[0], x_data[-1], 1000)
    y_fit = new_f(myoutput.beta, x_fit)
    
    chi_squared = chi_squared_function(x_data, y_data, y_err, myoutput.beta)
    reduced_chi = chi_squared / (len(x_data) - 2)
    print(f"Reduced chi squared = {reduced_chi}")

    plotter(x_data, y_data, x_fit, y_fit, x_err, y_err)

main()