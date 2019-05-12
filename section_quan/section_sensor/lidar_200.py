import pandas as pd ##dataread##
import matplotlib.pyplot as plt ##drawhist##
import numpy as np
import math  ##calcstddev
from scipy.stats import norm

sample_time = 10000
data = pd.read_csv("sensor_data_200.txt", delimiter=" ",
                   header=None, names=("date","time","ir","lidar"))
mean1 = sum(data['lidar'].values)/len(data['lidar'].values)
mean2 = data['lidar'].mean()

freqs = pd.DataFrame(data['lidar'].value_counts())
freqs['probs'] = freqs['lidar']/len(data['lidar'])

def create_simple_graph_from_ss_data():
    #print(data)
    #print(data["lidar"][0:5])
    
    #print(data['lidar'])
    #print(max(data["lidar"]))
    #print(min(data["lidar"]))

    data["lidar"].hist(bins=max(data["lidar"]) - min(data["lidar"]),align="left")

def change_noise_to_numerical_expression():
    print('mean1:', mean1)
    print('mean2:', mean2)

    #Draw mean into graph
    data['lidar'].hist(bins=max(data['lidar']) - min(data['lidar']), color='orange', align='left')
    plt.vlines(mean1, ymin=0, ymax=5000,color='red')

#####################################################################################
def calculate_variance_and_std_deviation():
    global stdev1,stdev2
    zs = data['lidar'].values
    mean = sum(zs)/len(zs)
    diff_square = [ (z - mean)**2 for z in zs]

    sampling_var = sum(diff_square)/(len(zs))   #標本分散
    unbiased_var = sum(diff_square)/(len(zs)-1) #不変分散

    print("標準偏差:",sampling_var)
    print("不変分散:", unbiased_var)

    #Pandas functions
    pandas_sampling_var = data['lidar'].var(ddof=False) #標本分散
    pandas_default_var = data['lidar'].var()  #デフォルト(不変分散)

    #print (pandas_sampling_var)
    #print (pandas_default_var)

    #Numpy functions
    numpy_default_var = np.var(data['lidar']) #デフォルト(標本分散)
    numpy_unbiased_var = np.var(data['lidar'], ddof=1)#不変分散

    #print (numpy_default_var)
    #print (numpy_unbiased_var)
    
    #定義から計算
    stdev1= math.sqrt(sampling_var)
    stdev2= math.sqrt(unbiased_var)
    
    #Use pandas
    pandas_stddev = data['lidar'].std()

    #print(stdev1)
    #print(stdev2)
    #print(pandas_stddev)
    
#####################################################################################
def get_freqs_probs():
    print(freqs)
    print('sum:', sum(freqs['probs']))

    #freqs['probs'].sort_index().plot.bar() ###probdist### probs histogram
    #freqs['probs'].hist(bins=len(freqs['probs']))

    samples = [drawing() for i in range(sample_time)]
    #samples = [drawing() for i in range(len(data))]
    simulated = pd.DataFrame(samples, columns=['lidar'])
    p = simulated['lidar']
    p.hist(bins=max(p) - min(p), color='orange',align='left')


def drawing(): # choosing something
    return freqs.sample(n=1, weights='probs').index[0]

#####################################################################################
def p(z, mu=209.7, dev=23.4):
    return math.exp(-(z-mu)**2/(2*dev))/math.sqrt(2*math.pi*dev)

def draw_conti_gauss_dis():
    zs = range(190,230)
    #ys = [p(z) for z in zs]
    ys = [norm.pdf(z,mean1,stdev1) for z in zs]
    plt.plot(zs,ys)

def prob(z, width=0.5): ###prob_plot_from_def###
    return width *(p(z-width) + p(z+width))

def draw_gauss_with_interger(): ###???????ask question####
    zs = range(190,230)
    ys = [prob(z) for z in zs]

    plt.bar(zs,ys, color='red', alpha=0.2)
    f=freqs['probs'].sort_index()
    plt.bar(f.index, f.values, color='blue', alpha=0.2)

def draw_cumul_dis(): #変数zが実数の時、確率密度分布pが積分して以下の図は累積分布関数とよばれる。
    zs = range(190,230)
    ys = [norm.cdf(z,mean1, stdev1) for z in zs]

    plt.plot(zs, ys, color='red')

def draw_prob_dis_from_cdf_diff():
    zs = range(190,230)
    ys = [norm.cdf(z+0.5, mean1, stdev1) - norm.cdf(z-0.5, mean1, stdev1) for z in zs]
    plt.bar(zs,ys)
    #plt.plot(zs,ys)

#####################################################################################
if __name__ == '__main__':
    #change_noise_to_numerical_expression()
    calculate_variance_and_std_deviation()
    get_freqs_probs()
    #print(drawing())
    #draw_conti_gauss_dis()
    #draw_gauss_with_interger()
    #draw_cumul_dis()
    #draw_prob_dis_from_cdf_diff()
    plt.show()
