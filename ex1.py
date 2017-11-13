# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import matrix_analysis

def main():
    system = matrix_analysis.System()
    system.load_nodes_csv('ex1_nodes.csv')
    system.load_members_csv('ex1_members.csv')

    system.solve()

    system.draw_displacement(disp_scale=1, x_margin_scale=2,y_margin_scale=1)
    system.print_result()


if __name__ == '__main__':
    main()
