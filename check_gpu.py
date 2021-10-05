import GPUtil
import psutil


def get_gpu_info():
    '''
    :return:
    '''
    Gpus = GPUtil.getGPUs()
    gpulist = []
    # GPUtil.showUtilization()
    for gpu in Gpus:
        l1 = "GPU_NAME = "+str(gpu.name)
        l2 = "GPU_TOTAL = "+str(gpu.memoryTotal)
        l3 = "GPU_USED = "+str(gpu.memoryUsed)
        l4 = "GPU_USE_PROPORTION = "+str(gpu.memoryUtil*100)
        l5 = "GPU_FREE = "+str(gpu.memoryFree)
        gpulist.append(l1)
        gpulist.append(l2)
        gpulist.append(l3)
        gpulist.append(l4)
        gpulist.append(l5)

    return gpulist

def write_info(file_name,gpu_info):
    with open(file_name, "w") as fd:
        fd.write("============GPU INFO=======\n")
        for i in gpu_info:
            fd.write(i+"\n")
        fd.write("=========CPU AND RAM=======\n")
        fd.write('RAM memory % used:{}'.format(psutil.virtual_memory()[2]))