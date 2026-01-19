import segyio
import numpy as np
import pyvista as pv

def read_segy_data_only(segy_path):
    """
    读取SEGY文件所有道数据，返回三维数组 (tracecount, 1, samples_per_trace)，不读取道头。
    """
    with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size

        data_2d = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            data_2d[i, :] = f.trace[i]

    # 扩展一个维度，变成3维数组 (tracecount, 1, samples)
    data_3d = data_2d[:, :]

    return data_3d

if __name__ == "__main__":
    segy_file = r"F3data.sgy"
    



    # 读取 SEG-Y 文件并加载数据到 dem
    dem = pv.read(segy_file)

    # 获取 dem 对象的维度 (通常是 nx, ny, nz)
    dimensions = dem.dimensions
    swapped_dimensions = (dimensions[1], dimensions[0], dimensions[2])
    print(swapped_dimensions) 

    data_3d = read_segy_data_only(segy_file)
    print("数据形状 (道数,  时间采样点数):", data_3d.shape)   
    if data_3d.shape != (swapped_dimensions[0]*swapped_dimensions[1], swapped_dimensions[2]):
        print("警告：读取的数据形状与维度不匹配！")
    np.save("F3data2.npy", data_3d)