# -*- coding: utf-8 -*-

import matrix_analysis

def main():
    system = matrix_analysis.System()
    system.load_nodes_csv('ex2_nodes.csv')
    system.load_members_csv('ex2_members.csv')

    system.solve()

    system.print_result()
    system.draw_displacement(disp_scale=10, x_margin_scale=0.1,y_margin_scale=0.5)


if __name__ == '__main__':
    main()
