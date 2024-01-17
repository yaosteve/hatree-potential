# 导入NumPy库，用于数值计算
import numpy as np

# 定义一个函数，用于从CHGCAR文件中读取电子电荷密度
def read_chgcar(CHGCAR):
    # filename是CHGCAR文件的路径
    # 返回一个三维数组，表示电子电荷密度

    # 打开文件
    with open(CHGCAR, "r") as f:
        # 跳过前六行，不需要的信息
        for _ in range(6):
            f.readline()

        # 读取第七行，得到网格的大小
        nx, ny, nz = map(int, f.readline().split())

        # 跳过后面的行，直到找到电荷密度的开始
        while True:
            line = f.readline()
            if len(line.split()) == nx:
                break

        # 定义一个空数组，用于存储电荷密度
        rho = np.zeros((nx, ny, nz))

        # 用循环遍历每个网格点，读取电荷密度的值
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # 如果一行的数据读完了，就读取下一行
                    if i % 5 == 0:
                        line = f.readline()
                        data = line.split()
                    # 从数据中取出对应的值，转换为浮点数，存入数组
                    try:
                        rho[i, j, k] = float(data[i % 5])
                    except: 
                        IndexError
                    print(f"IndexError: list index out of range at {i}, {j}, {k}")
                    continue

    # 返回电荷密度数组
    return rho

# 定义一个函数，用于从POSCAR文件中读取晶格参数和体积
def read_poscar(POSCAR):
    # filename是POSCAR文件的路径
    # 返回一个三维数组，表示晶格参数，和一个浮点数，表示体积

    # 打开文件
    with open(POSCAR, "r") as f:
        # 跳过第一行，不需要的信息
        f.readline()

        # 读取第二行，得到晶格的缩放因子
        scale = float(f.readline())

        # 读取第三到第五行，得到晶格的向量
        a1 = np.array(list(map(float, f.readline().split())))
        a2 = np.array(list(map(float, f.readline().split())))
        a3 = np.array(list(map(float, f.readline().split())))

        # 将晶格向量乘以缩放因子，得到真实的晶格参数
        a1 = a1 * scale
        a2 = a2 * scale
        a3 = a3 * scale

        # 将晶格参数组合成一个三维数组
        lattice = np.array([a1, a2, a3])

        # 计算晶格的体积，用向量积的方法
        volume = np.dot(a1, np.cross(a2, a3))

    # 返回晶格参数和体积
    return lattice, volume

# 定义一个函数，用于计算哈特里势
def hartree_potential(rho, lattice, volume):
    # rho是电子电荷密度，是一个三维数组
    # lattice是晶格参数，是一个三维数组
    # volume是晶格的体积，是一个浮点数
    # 返回一个三维数组，表示哈特里势

    # 定义一个常数，表示4π
    four_pi = 4 * np.pi

    # 定义一个常数，表示真空介电常数，单位是法拉/米
    epsilon_0 = 8.8541878128e-12

    # 定义一个常数，表示元电荷，单位是库伦
    e = 1.602176634e-19

    # 得到电荷密度的网格的大小
    nx, ny, nz = rho.shape

    # 定义一个空数组，用于存储哈特里势
    v_h = np.zeros_like(rho)

    # 用循环遍历每个网格点
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 计算当前点的电荷密度
                ni = rho[i, j, k]

                # 如果电荷密度为零，那么哈特里势也为零，跳过这个点
                if ni == 0:
                    continue

                # 计算当前点的坐标，用晶格参数的线性组合
                ri = i * lattice[0] / nx + j * lattice[1] / ny + k * lattice[2] / nz

                # 计算当前点的哈特里势，根据泊松方程的积分形式
                # V_H(r) = 1 / (4πε_0) ∫ ρ(r') / |r - r'| dV'
                # 用数值积分的方法，对所有的网格点求和
                for i2 in range(nx):
                    for j2 in range(ny):
                        for k2 in range(nz):
                            # 计算另一个点的电荷密度
                            n2 = rho[i2, j2, k2]

                            # 如果电荷密度为零，那么对积分没有贡献，跳过这个点
                            if n2 == 0:
                                continue

                            # 计算另一个点的坐标，用晶格参数的线性组合
                            r2 = i2 * lattice[0] / nx + j2 * lattice[1] / ny + k2 * lattice[2] / nz

                            # 计算两点之间的距离，用欧几里得范数
                            dr = np.linalg.norm(ri - r2)

                            # 计算积分的被积函数的值，除以两点之间的距离
                            f = n2 / dr

                            # 计算积分的微元，用晶格的体积除以网格的总数
                            dv = volume / (nx * ny * nz)

                            # 计算积分的和，乘以常数因子
                            v_h[i, j, k] += f * dv * e / (four_pi * epsilon_0)

    # 返回哈特里势数组
    return v_h

# 定义一个函数，用于将哈特里势输出成CHGCAR的格式
def write_chgcar(v_h, lattice, VHGCAR):
    # v_h是哈特里势，是一个三维数组
    # lattice是晶格参数，是一个三维数组
    # filename是输出文件的路径

    # 打开文件
    with open(VHGCAR, "w") as f:
        # 写入第一行，任意字符串
        f.write("Hartree potential\n")

        # 写入第二行，晶格的缩放因子，设为1
        f.write("1.0\n")

        # 写入第三到第五行，晶格的向量
        for i in range(3):
            f.write(" ".join(map(str, lattice[i])) + "\n")

        # 写入第六行，原子的种类，设为1
        f.write("1\n")

        # 写入第七行，原子的个数，设为0
        f.write("0\n")

        # 写入第八行，原子的坐标模式，设为Cartesian
        f.write("Cartesian\n")

        # 写入第九行，网格的大小
        nx, ny, nz = v_h.shape
        f.write(" ".join(map(str, (nx, ny, nz))))

        # 定义CHGCAR文件的路径
        chgcar_file = "CHGCAR"

        # 定义POSCAR文件的路径
        poscar_file = "POSCAR"

        # 调用函数，读取电子电荷密度
        rho = read_chgcar(chgcar_file)

        # 调用函数，读取晶格参数和体积
        lattice, volume = read_poscar(poscar_file)

        # 调用函数，计算哈特里势
        v_h = hartree_potential(rho, lattice, volume)

        # 定义输出文件的路径
        output_file = "VHGCAR"

        # 调用函数，将哈特里势输出成CHGCAR的格式
        write_chgcar(v_h, lattice, output_file)

