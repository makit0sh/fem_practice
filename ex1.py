# -*- coding: utf-8 -*-

import matrix_analysis

def main():
    system = matrix_analysis.System()
    system.load_nodes_csv('ex1_nodes.csv')
    system.load_members_csv('ex1_members.csv')

    system.solve()

    system.print_result()
    system.draw_displacement(disp_scale=100)


if __name__ == '__main__':
    main()
