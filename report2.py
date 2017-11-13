# -*- coding: utf-8 -*-
"""数値構造解析課題2の計算用スクリプト"""

import matrix_analysis

def main():
    """main関数。トラスのシステムを読み込み，計算結果を表示する"""
    system = matrix_analysis.System()
    system.load_nodes_csv('report2_nodes.csv')
    system.load_members_csv('report2_members.csv')

    system.solve()

    system.draw_displacement(y_margin_scale=0.5)
    system.print_result()


if __name__ == '__main__':
    main()
