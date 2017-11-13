# -*- coding: utf-8 -*-
"""Matrix構造解析を行うモジュール.

二次元トラス構造の解析用.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

class Node:
    """節点データのクラス.

    Attributes:
        number (int): 節点の番号
        index (int): 計算用のindex, 0からスタート
        x (double): 全体座標系でのx座標
        y (double): 全体座標系でのy座標
        disp_cond_x (double): x方向変位境界条件.初期値は, None
        disp_cond_y (double): y方向変位境界条件.初期値は, None
        load_cond_x (double): x方向荷重境界条件.初期値は, None
        load_cond_y (double): y方向荷重境界条件.初期値は, None
    """
    def __init__(self, number_param, index_param, x_param, y_param, disp_cond_x_param, disp_cond_y_param, load_cond_x_param, load_cond_y_param):
        """節点のコンストラクタ.

        Args:
            number_param (int): 節点の番号
            index_param (int): 計算用のindex, 0からスタート
            x_param (double): 全体座標系でのx座標
            y_param (double): 全体座標系でのy座標
            disp_cond_x_param (float): 変位境界条件.何もないときは, None
            disp_cond_x_param (float): 変位境界条件.何もないときは, None
            load_cond_y_param (float): 荷重境界条件.何もないときは, None
            load_cond_y_param (float): 荷重境界条件.何もないときは, None
        """
        self.number = number_param
        self.index = index_param
        self.x = x_param
        self.y = y_param
        self.disp_cond_x = disp_cond_x_param
        self.disp_cond_y = disp_cond_y_param
        self.load_cond_x = load_cond_x_param
        self.load_cond_y = load_cond_y_param

    def __str__(self):
        expl = 'num: '+str(self.number)+', x: '+str(self.x)+', y: '+str(self.y)+', disp_cond_x: '+str(self.disp_cond_x)+', disp_cond_y: '+str(self.disp_cond_y)+', load_cond_x: '+str(self.load_cond_x)+', load_cond_y: '+str(self.load_cond_y)
        return expl

    def has_disp_cond(self):
        """変位境界条件を持っているかどうかを返す.

        Returns:
            disp_cond が存在するならTrue
        """
        if self.disp_cond_x is not None or self.disp_cond_y is not None:
            return True
        else:
            return False

    def has_load_cond(self):
        """荷重境界条件を持っているかどうかを返す.

        Returns:
            load_cond_cond が存在するならTrue
        """
        if self.load_cond_x is not None or self.load_cond_y is not None:
            return True
        else:
            return False

    def set_load_cond_x(self, Px):
        """x方向の荷重をセットする.

        Args:
            Px (double): x方向の荷重条件.
        """
        if self.has_disp_cond():
            raise Exception('one node can only have one condition')
        self.load_cond_x = Px

    def set_load_cond_y(self, Py):
        """y方向の荷重をセットする.

        Args:
            Py (double): y方向の荷重条件.
        """
        if self.has_disp_cond():
            raise Exception('one node can only have one condition')
        self.load_cond_y = Py

    def add_load_cond_x(self, Px):
        """x方向の荷重を, 現在の条件に加える.

        Args:
            Px (double): 追加するx方向荷重条件.
        """
        if self.has_disp_cond():
            raise Exception('one node can only have one condition')
        self.load_cond_x += Px

    def add_load_cond_y(self, Py):
        """y方向の荷重を, 現在の条件に加える.

        Args:
            Px (double): 追加するx方向荷重条件.
        """
        if self.has_disp_cond():
            raise Exception('one node can only have one condition')
        self.load_cond_y += Py

class Member:
    """部材データのクラス.

    Attributes:
        nodes (list(Node)): 関連する節点のリスト
        area (double): 断面積
        E (double): ヤング率
    """
    def __init__(self, nodes_param, area_param, E_param):
        """コンストラクタ.

        Args:
            nodes_param (list(int)): 関連する節点の番号のリスト
            area_param (double): 断面積
            E_param (double): ヤング率
        """
        self.nodes = nodes_param
        self.area = area_param
        self.E = E_param

    def __str__(self):
        expl = 'nodes: '+str(self.nodes)+', area: '+str(self.area)+', E: '+str(self.E)
        return expl

class Load:
    """荷重データのクラス.

    Attributes:
        node (int): 荷重がかかっている節点の番号
        x (double): x方向の荷重
        y (double): y方向の荷重
    """
    def __init__(self, node_num, Px, Py):
        """コンストラクタ.

        Args:
            node_num (int): 荷重がかかっている節点の番号
            Px (double): x方向の荷重.
            Py (double): y方向の荷重
        """
        self.node = node_num

        self.x = Px
        self.y = Py

    def __str__(self):
        expl = 'node: '+str(self.node)+', Px: '+str(self.x)+', Py: '+str(self.y)
        return expl

class System:
    """構造解析を行う対象のシステム.

    Attributes:
        nodes (dict(Node)): システム内に存在する節点の辞書.辞書のキーがnode番号と対応する.
        members (dict(Member)): システム内に存在する部材の辞書.辞書のキーが部材の番号に対応する.
        K_total (numpy.array): トータルの剛性マトリックス
        f_total (numpy.array): トータルの荷重ベクトル
        d_total (numpy.array): トータルの変位ベクトル
    """
    def __init__(self):
        """Attributesを初期化するコンストラクタ
        """
        self.nodes = dict()
        self.members = dict()
        self.K_total = None

    def load_nodes_csv(self, filename):
        """節点の条件をcsvファイルから読み取る.

        Note:
            csvファイルは, 一行目はヘッダー, 二行目以降にデータ(番号, x座標, y座標, x方向変位境界条件, y方向変位境界条件, x方向荷重条件, y方向荷重条件(いずれもない場合はNone))

        Args:
            filename (str): ファイル名
        """
        current_index = 0
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                if len(row) == 7:
                    try:
                        dx = float(row[3])
                    except ValueError:
                        dx = None
                    try:
                        dy = float(row[4])
                    except ValueError:
                        dy = None
                    try:
                        Px = float(row[5])
                    except ValueError:
                        Px = None
                    try:
                        Py = float(row[6])
                    except ValueError:
                        Py = None
                    self.nodes[int(row[0])] = Node(int(row[0]), current_index, float(row[1]), float(row[2]), dx, dy, Px, Py)
                    current_index += 1
                else:
                    raise Exception('invalid data input')

    def load_members_csv(self, filename):
        """ 節点の条件をcsvファイルから読み取る.

        Note:
            csvファイルは，一行目はヘッダー，二行目以降にデータ（番号，右端の節点番号，左端の節点番号，断面積，ヤング率)

        Args:
            filename (str): ファイル名
        """
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                if len(row) == 5:
                    self.members[int(row[0])] = Member([int(row[1]), int(row[2])], float(row[3]), float(row[4]))
                else:
                    raise Exception('invalid data input')

    def calculate_partial_stiffness_matrix(self):
        """各部材に対して, 全体座標系で表した剛性マトリックスを計算する."""
        for m in self.members.values():
            node = [self.nodes[m.nodes[0]], self.nodes[m.nodes[1]]]
            L = np.sqrt((node[1].x-node[0].x)**2 + (node[1].y-node[0].y)**2)
            c = (node[1].x - node[0].x) / L
            s = (node[1].y - node[0].y) / L
            m.K = np.array([[c*c, c*s, -c*c, -c*s],
                            [c*s, s*s, -c*s, -s*s],
                            [-c*c, -c*s, c*c, c*s],
                            [-c*s, -s*s, c*s, s*s]]) * m.E * m.area / L

    def calculate_total_stiffness_matrix(self):
        """全体の剛性マトリックスを計算する"""
        matrix_size = 2*len(self.nodes)
        self.K_total = np.zeros((matrix_size, matrix_size))
        for m in self.members.values():
        #for m in [self.members[1]]:
            node = [self.nodes[m.nodes[0]], self.nodes[m.nodes[1]]]
            self.K_total[node[0].index*2:node[0].index*2+2, node[0].index*2:node[0].index*2+2] += m.K[0:2, 0:2]
            self.K_total[node[0].index*2:node[0].index*2+2, node[1].index*2:node[1].index*2+2] += m.K[0:2, 2:4]
            self.K_total[node[1].index*2:node[1].index*2+2, node[0].index*2:node[0].index*2+2] += m.K[2:4, 0:2]
            self.K_total[node[1].index*2:node[1].index*2+2, node[1].index*2:node[1].index*2+2] += m.K[2:4, 2:4]

        self.f_total = np.zeros(matrix_size)
        self.d_total = np.zeros(matrix_size)
        for node in self.nodes.values():
            self.f_total[node.index*2] = node.load_cond_x
            self.f_total[node.index*2+1] = node.load_cond_y
            self.d_total[node.index*2] = node.disp_cond_x
            self.d_total[node.index*2+1] = node.disp_cond_y

    def solve_stiffness_equation(self):
        """剛性方程式を解く"""
        Kff_sf = self.K_total[np.where(np.logical_not(np.isnan(self.f_total))) ]
        Kff_size = Kff_sf.shape[0]
        Kff = Kff_sf[ :,  np.where(np.logical_not(np.isnan(self.f_total)))].reshape(Kff_size, Kff_size)
        Ksf = Kff_sf[:, np.where(np.isnan(self.f_total))].reshape(Kff_size, Kff_sf.shape[1]-Kff_size)

        Kfs_ss = self.K_total[np.where(np.logical_not(np.isnan(self.d_total)))]
        Kss_size = Kfs_ss.shape[0]
        Kfs = Kfs_ss[:, np.where(np.isnan(self.d_total))].reshape(Kss_size, Kfs_ss.shape[1]-Kss_size)
        Kss = Kfs_ss[:, np.where(np.logical_not(np.isnan(self.d_total)))].reshape(Kss_size, Kss_size)

        ff = self.f_total[np.where(np.logical_not(np.isnan(self.f_total)))]
        fs = self.f_total[np.where(np.isnan(self.f_total)) ]

        df = self.d_total[np.where(np.isnan(self.d_total)) ]
        ds = self.d_total[np.where(np.logical_not(np.isnan(self.d_total))) ]

        df = np.linalg.inv(Kff).dot(ff - Ksf.dot(ds))
        fs = Kfs.dot(df) + Kss.dot(ds)

        self.f_total[np.where(np.isnan(self.f_total)) ] = fs
        self.d_total[np.where(np.isnan(self.d_total)) ] = df

    def solve(self):
        """剛性方程式をつくり, 解を求める"""
        self.calculate_partial_stiffness_matrix()
        self.calculate_total_stiffness_matrix()
        self.solve_stiffness_equation()

    def draw_displacement(self, disp_scale=10, x_margin_scale=0.1, y_margin_scale=0.1):
        """トラスの変形の様子を描画する
        
        Args:
            disp_scale (int): 変位の拡大倍率. default=10
            x_margin_scale (int): x方向の余白の拡大倍率. default=0.1
            y_margin_scale (int): y方向の余白の拡大倍率. default=0.1
        """
        # 元のsystemを表示
        for n in self.nodes.values():
            plt.plot(n.x, n.y, 'ko')
        for member in self.members.values():
            node = [[self.nodes[member.nodes[0]].x, self.nodes[member.nodes[1]].x], 
                    [self.nodes[member.nodes[0]].y, self.nodes[member.nodes[1]].y]]
            plt.plot(node[0], node[1], 'k-', lw=2)

        # 変形後のシステムを表示
        sorted_node = sorted(self.nodes.values(), key=lambda n: n.index)
        node_xy = []
        for n in sorted_node:
            node_xy.append(n.x)
            node_xy.append(n.y)
        node_xy = np.array(node_xy)
        node_xy += self.d_total*disp_scale
        for i in range(0, node_xy.size, 2):
            plt.plot(node_xy[i], node_xy[i+1], 'bo')
        for member in self.members.values():
            node = [[self.nodes[member.nodes[0]].x+(self.d_total*disp_scale)[self.nodes[member.nodes[0]].index*2],
                    self.nodes[member.nodes[1]].x+(self.d_total*disp_scale)[self.nodes[member.nodes[1]].index*2] ], 
                    [self.nodes[member.nodes[0]].y+(self.d_total*disp_scale)[self.nodes[member.nodes[0]].index*2+1], 
                    self.nodes[member.nodes[1]].y+(self.d_total*disp_scale)[self.nodes[member.nodes[1]].index*2+1] ]]
            plt.plot(node[0], node[1], 'b-', lw=2)

        #荷重を矢印で表示
        node_x = node_xy[0::2]
        node_y = node_xy[1::2]
        f_x = self.f_total[0::2]
        f_y = self.f_total[1::2]
        plt.quiver(node_x, node_y, f_x, f_y)

        #整形して表示
        plt.axis('scaled')
        xmin = min([n.x for n in self.nodes.values()])
        xmax = max([n.x for n in self.nodes.values()])
        xmargin = (xmax-xmin)*x_margin_scale
        plt.xlim(xmin-xmargin, xmax+xmargin)
        ymin = min([n.y for n in self.nodes.values()])
        ymax = max([n.y for n in self.nodes.values()])
        ymargin = (ymax-ymin)*y_margin_scale
        plt.ylim(ymin-ymargin, ymax+ymargin)
        plt.show()

    def calculate_stress(self):
        """各部材の応力を計算する
        
        Returns:
            stress (dict(float)): 部材の番号をキーとした各部材の応力の辞書
        """
        stress = dict()
        for i, m in zip(self.members.keys(),self.members.values()):
            node = [self.nodes[m.nodes[0]], self.nodes[m.nodes[1]]]
            L = np.sqrt((node[1].x-node[0].x)**2 + (node[1].y-node[0].y)**2)
            dL = np.sqrt((node[1].x+self.d_total[node[1].index*2]-node[0].x-self.d_total[node[0].index*2])**2 + 
                    (node[1].y+self.d_total[node[1].index*2+1]-node[0].y-self.d_total[node[0].index*2+1])**2 ) -L
            stress[i] =  m.E* dL/L
        return stress

    def print_result(self):
        """計算結果を表示する"""
        sorted_node = sorted(self.nodes.values(), key=lambda n: n.index)
        print('各ノードにかかる荷重')
        for n in sorted_node:
            i=n.index
            print('node no. '+str(n.number)+ ' : '+ str([self.f_total[i], self.f_total[i+1]]))

        print('各ノードの変位')
        for n in sorted_node:
            i=n.index
            print('node no. '+str(n.number)+ ' : '+ str([self.d_total[i], self.d_total[i+1]]))

        print('各部材内部の応力')
        stress = self.calculate_stress()
        for i, s in zip(stress, stress.values()):
            print('member no. '+str(i)+' : '+str(s))

        print('荷重f')
        print(self.f_total)
        print('全体剛性マトリックス')
        print(self.K_total)
        print('変位d')
        print(self.d_total)

        print('f と Kd の計算誤差')
        print(self.f_total-self.K_total.dot(self.d_total))
