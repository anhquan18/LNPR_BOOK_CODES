# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

global freqs, probs

data = pd.read_csv("sensor_data_600.txt", delimiter=" ",
                   header=None,names=("date","time","ir","lidar"))
data["hour"] = [e//10000 for e in data.time]
data['hour'] = [e//10000 for e in data.time]
d = data.groupby("hour")
#print(data)
#data.lidar.plot() #plot histogram of lidar value with time
#data.ir.plot()   #plot histogram of ir value with time

each_hour = {i: d.lidar.get_group(i).value_counts().sort_index() for i in range(24)} # create 24 DataFrame objects
#print(each_hour)
freqs = pd.concat(each_hour, axis=1) # use concat for continues
#print(freqs)
freqs = freqs.fillna(0) # Fill value isn't exist (NaN) with 0
# P(z,t) 確率の乗法定理
probs = freqs/len(data) # Calculate probabilistic of all value at a certain time condition

#以下は確率の乗法定理と確率条件付きとの関係を表す
#P(x,y) = P(x|y)*P(y) = P(x)*P(y|x)

#print(freqs)
#print('\n')
#print('probs')
#print(probs)
#print(probs.transpose())

#print(pd.DataFrame(probs.transpose()))
p_z = pd.DataFrame(probs.transpose().sum()) #from probs of sensor value at each hour -> use sum to Calculate the probs of each sensor value
p_t = pd.DataFrame(probs.sum()) # P(t)

#print (p_t[0])

#print(p_z)
#print('\n')
#print(p_t)
#print('\n')
#print(p_t[0])

cond_z_t = probs/p_t[0]  # P(z|t) == cond_z_t == P(z,t)/P(t)

cond_t_z = probs.transpose()/probs.transpose().sum() #P(t|z) == P(t,z)/P(t)
#print(cond_t_z)


# HANDLE DAtA
#######################################################################################
def make_his_from_600():
    data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align='left')

def group_hour_data():
    #d.lidar.mean().plot()
    # making histogram graph of hour 6 and hour 14
    #print(d.get_group(6))
    #print(d.lidar.get_group(6)) # all of value from sensor at hour 6
    d.lidar.get_group(6).hist() # make hist from all of value the sensor got at hour 6
    d.lidar.get_group(14).hist()
    d.lidar.get_group(8).hist()

def p_t_z_by_bayes():
    print("P(z=630) = ", p_z[0][630])
    print("P(t=13) = ", p_t[0][13])
    print("P(t=13|z=630) = ", cond_t_z[630][13])
    print("Bayes(z=630|t=13) = ", cond_t_z[630][13]*p_z[0][630]/p_t[0][13])

    print("answer P(z=630|t=13) = ", cond_z_t[13][630])

def bayes_estimation(sensor_value, current_estimation):
    new_estimation = []
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value]*current_estimation[i]) #P(z = sensor_value|t=0->23) * P(t=0->23)
    return new_estimation/sum(new_estimation) #正規化P(t|z) = (P(z=sensor_value|t) * P(t=0)) / sigma P(z= sensor_values ,t=0->23)

#######################################################################################


# CREATE GRAPH FROM DATA
#######################################################################################
def condition_probs():
    print('heatmap1')
    plt.figure(1)
    sb.heatmap(probs) # generate heatmap for probs of each hour
    plt.figure(2)
    print('heatmap2')
    sb.jointplot(data["hour"], data["lidar"], data, kind='kde') # another heatmap

def prob_of_t():
    p_t.plot()
    #p_t.transpose() 
    print(p_t)
    print('sum of p(t):', p_t.sum())

def prob_of_z():
    p_z.plot()
    print(p_z)

def probs_of_z_with_certain_t():
    print(cond_z_t)
    cond_z_t[6].plot.bar(color='blue',alpha=0.5)
    cond_z_t[14].plot.bar(color='orange',alpha=0.5)

def graph_by_bayes_estimation(*sensor_values):
    estimation = p_t[0]
    for value in sensor_values:
        estimation = bayes_estimation(value, estimation) # from multi sensors value and P(t=0->23)
    plt.plot(estimation)
#######################################################################################

if __name__ == "__main__":
    #make_his_from_600()
    #group_hour_data()
    #condition_probs()
    #prob_of_t()
    #prob_of_z()
    #probs_of_z_with_certain_t()
    #p_t_z_by_bayes()
    #graph_by_bayes_estimation(630)
    #graph_by_bayes_estimation(630,632,636)
    graph_by_bayes_estimation(617,624,619)
    plt.show()
