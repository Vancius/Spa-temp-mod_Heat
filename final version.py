import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.linalg import expm
import time
import imageio

time_start=time.time()

class setup:
    def __init__(self,sys_type="wave driving both",equ_type="rho in",sign=1,temp_gif_save=False,avg_save=False,
                 T1=393,T2=293,rho0=1000.,kappa0=100.,
                 amp_kappa=.99,amp_rho=1/3,
                 num_space_points=100,space_step=.01,num_space_periods=10,
                 velocity=1.,num_time_periods=150,points_per_time_period=500):
        self.sys_type = sys_type  # can be: "time independent", "wave driving both", "varying only rho", "varying only kappa"
        self.equ_type=equ_type  # can be :"rho in" or "rho out".
        self.sign=sign   # the direction of wave driving(sign=1 means to the right, sign=-1 means to the left)
        self.temp_save_choice=temp_gif_save  # save the temperature gif or not
        self.avg_save_choice=avg_save # save the avg_current as txt or not
        self.T1,self.T2,self.rho0,self.kappa0=T1,T2,rho0,kappa0
        self.amp_kappa,self.amp_rho=amp_kappa,amp_rho
        self.N,self.space_step,self.num_space_periods=num_space_points,space_step,num_space_periods
        self.velocity,self.num_time_periods=velocity,num_time_periods
        self.points_per_time_period=points_per_time_period*np.int(np.ceil(self.kappa0/100.))

        self.len_space_period=self.space_step*self.N/self.num_space_periods
        self.len_time_period=self.len_space_period/np.abs(self.velocity)
        self.time_step=self.len_time_period/self.points_per_time_period

        self.kappa=self.kappa0*np.ones((self.N-1,1))
        self.rho=self.rho0*np.ones((self.N,1))
        self.rho[0],self.rho[-1]=np.inf,np.inf

        self.temp=self.T2*np.ones((self.N,1))
        self.temp[0]=self.T1

        self.rho_T = self.rho * self.temp

        self.temp_to_j=self.fun_temp_to_j(self.kappa)
        self.j_to_rhoT=self.fun_j_to_rhoT()

        self.moment_current=np.dot(self.temp_to_j, self.temp)
        self.avg_current=np.zeros((self.N-1,1))
        self.avg_temp = np.zeros((self.N , 1))

        self.num_temp_hist_points = 10 * self.points_per_time_period
        self.temp_his = np.zeros((self.N, self.num_temp_hist_points))


    def fun_temp_to_j(self,kappa):
        result=np.zeros((self.N-1,self.N))
        i=np.array(range(self.N-1))
        result[i,i]=-kappa[i,0]
        result[i,i+1]=kappa[i,0]
        result/=(-self.space_step)
        return result

    def fun_j_to_rhoT(self):
        result=np.zeros((self.N,self.N-1))
        i=np.array(range(self.N-1))
        result[i,i]=1
        result[i+1,i]=-1
        result/=(-self.space_step)
        return result

    def kappa_t(self,t):
        # generating kappa at time t, and renewing the matrix temp_to_j
        i = np.array(range(self.N - 1))
        self.kappa[i,0] = self.kappa0 * (1+self.amp_kappa*np.sin(2*np.pi*(+(i+1/2)*self.space_step/self.len_space_period-self.sign*
                                                                        t/self.len_time_period)))
        # NOTE: the 1/2 in the previous formula is essential, since the current points are in the middle of temperature points. So 1/2
        # preserves the reflection symmetry of space.
        self.temp_to_j = self.fun_temp_to_j(self.kappa)

    def rho_t(self,t):
        i = np.array(range(self.N - 2)) + 1
        self.rho[i,0]=self.rho0*(1+self.amp_rho*np.sin(2*np.pi*(+i*self.space_step/self.len_space_period-self.sign*
                                                                t/self.len_time_period)))

    def save_temp_gif(self):
        ims = []
        fig = plt.figure()
        for i in range(a.num_temp_hist_points):
            if i % (max(self.points_per_time_period/10,10)) == 0:
                y = np.linspace(0, 0, a.N)
                y[:] = a.temp_his[:, i]
                im = plt.plot(np.array(range(a.N)) * a.space_step, y)
                plt.title("rho:" + str(a.rho0) + "/kappa:" + str(a.kappa0))
                plt.xlabel("position/m")
                plt.ylabel("temperature/K")
                im = np.array(im)
                ims.append(im)
        ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
        ani.save(a.equ_type + "_rho_"+str(self.rho0)+"_kappa_"+str(self.kappa0)+" _temperature.html", writer='pillow')
        # the above saves the movie of temperature evolution

    def save_avg(self):
        fp=open("avg.txt","w")
        for i in range(self.N-1):
            if i==0:
                fp.write(self.equ_type+"\n")
                fp.write("rho_"+str(self.rho0)+"/kappa_"+str(self.kappa0)+"/T1_"+str(self.T1)+"/T2_"+str(self.T2)+"\n")
                fp.write("avg_temp"+"\t"+"avg_current"+"\n")
            fp.write(str(self.avg_temp[i,0])+"\t")
            fp.write(str(self.avg_current[i,0])+"\n")

    def evolve(self):
        if self.sys_type=="time independent":
            rho_drive,kappa_drive=False,False
        elif self.sys_type=="varying only rho":
            rho_drive,kappa_drive=True,False
        elif self.sys_type=="varying only kappa":
            rho_drive, kappa_drive = False, True
        elif self.sys_type=="wave driving both":
            rho_drive, kappa_drive = True, True
        else:
            print("no driving protocol")
            return 1

        for n in range(self.num_time_periods*self.points_per_time_period):
            # getting to steady state
            if kappa_drive==True:
                self.kappa_t(n * self.time_step)
            else:
                self.kappa_t(0)
            if rho_drive==True:
                self.rho_t(n * self.time_step)
            else:
                self.rho_t(0)

            i = np.array(range(self.N - 2)) + 1
            self.moment_current = np.dot(self.temp_to_j, self.temp)

            if self.equ_type=="rho in":
                self.rho_T += (np.dot(self.j_to_rhoT, self.moment_current) * self.time_step)
                self.temp[i, 0] = self.rho_T[i, 0] / self.rho[i, 0]
            elif self.equ_type=="rho out":
                del_rho_T = np.dot(self.j_to_rhoT, self.moment_current) * self.time_step
                self.temp[i, 0] = self.temp[i, 0] + del_rho_T[i, 0] / self.rho[i, 0]
            else:
                print("wrong equation type")
                return 1


        for n in range(self.num_time_periods * self.points_per_time_period):
            # the main driving sequence, averaged current is calculated
            m=self.num_time_periods * self.points_per_time_period+n
            i = np.array(range(self.N - 2)) + 1
            if kappa_drive==True:
                self.kappa_t(m * self.time_step)
            if rho_drive==True:
                self.rho_t(m * self.time_step)
            self.moment_current = np.dot(self.temp_to_j, self.temp)

            self.avg_current+=(self.moment_current/self.points_per_time_period/self.num_time_periods)
            self.avg_temp += (self.temp / self.points_per_time_period / self.num_time_periods)


            if self.equ_type=="rho in":
                self.rho_T += (np.dot(self.j_to_rhoT, self.moment_current) * self.time_step)
                self.temp[i, 0] = self.rho_T[i, 0] / self.rho[i, 0]
            elif self.equ_type=="rho out":
                del_rho_T = np.dot(self.j_to_rhoT, self.moment_current) * self.time_step
                self.temp[i, 0] = self.temp[i, 0] + del_rho_T[i, 0] / self.rho[i, 0]
            else:
                print("wrong equation type")
                return 1

            if self.num_time_periods * self.points_per_time_period-n<=self.num_temp_hist_points:
                self.temp_his[:,self.num_temp_hist_points-self.num_time_periods * self.points_per_time_period+n]=self.temp[:,0]

        if self.temp_save_choice==True:
            self.save_temp_gif()  # saving temperature evolution gif
        if self.avg_save_choice==True:
            self.save_avg()   #saving average temperature and current









#a=setup(T1=393,T2=293,sign=1,rho0=1000.,kappa0=20.,equ_type="rho in")
#a.evolve()




#b=setup(T1=293,T2=393,sign=1,equ_type="rho out")
#b.evolve()

"""
plt.figure()
plt.plot(a.avg_temp)
plt.show()

plt.figure()
plt.plot(a.avg_current)
plt.show()

print("avg current:",np.mean(a.avg_current))
print("current fluc:", np.abs((np.max(a.avg_current)-np.min(a.avg_current))/np.mean(a.avg_current)))
"""

"""
for i in range(a.num_temp_hist_points):
    if i%50==0:
        y = np.linspace(0, 0, a.N)
        y[:] = a.temp_his[:, i]
        plt.plot(np.array(range(a.N)) * a.space_step, y)
        plt.title("rho:" + str(a.rho0) + "/kappa:" + str(a.kappa0))
        plt.xlabel("position/m")
        plt.ylabel("temperature/K")
        plt.pause(.1)
        plt.cla()
plt.show()
"""








"""
plt.figure()
plt.plot(a.avg_temp,label="to right")
plt.plot(b.avg_temp,label="to left")
plt.legend()
plt.show()
"""

"""
print("to right:")
print("avg current:",np.mean(a.avg_current))
print("current fluc:", np.abs((np.max(a.avg_current)-np.min(a.avg_current))/np.mean(a.avg_current)))
print("to left:")
print("avg current:",np.mean(b.avg_current))
print("current fluc:",np.abs((np.max(b.avg_current)-np.min(b.avg_current))/np.mean(b.avg_current)))
"""














n=15
left_temp=np.linspace(193,493,n)
right_temp=np.linspace(193,493,n)

avg_current=np.zeros((n,n))
avg_current_error=np.zeros((n,n))
for i in range(n):
    for j in range(n):
        a=setup(T1=left_temp[i],T2=right_temp[j],equ_type="rho in",sys_type="time independent")
        a.evolve()
        avg_current[i,j]=np.mean(a.avg_current)
        avg_current_error[i,j]=np.abs(np.max(a.avg_current)-np.min(a.avg_current))/np.abs(np.mean(a.avg_current))

fp=open("avg_current.txt","w")
for i in range(n):
    for j in range(n):
        fp.write(str(left_temp[i]))
        fp.write("\t")
        fp.write(str(right_temp[j]))
        fp.write("\t")
        fp.write(str(avg_current[i,j]))
        fp.write("\t")
        fp.write(str(avg_current_error[i,j]))
        fp.write("\n")

fp.close()
















time_end=time.time()
print("time consumed: ", time_end-time_start)