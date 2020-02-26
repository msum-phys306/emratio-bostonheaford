import numpy as np

def HH_field(N,I,L):
    """
    Determine magnetic field of a Helmholtz pair.

    Arguments:
    N -- the number of loops wrapped on a coil
    I --  the current through the coils, in Amps
    L --the radius of a coil, or the separation between the two sides.
    

    Return
    float Bfield--magnetic field strength in Teslas


    """
    
    mu = 4 * np.pi * 10**(-7)
    
    B_field = ( 8 * mu * N * I) / ( ( 125**(1/2 ) ) * L )
    
    return B_field
    

####################################################
def Lorentz_force(q,v,B):
    """
    Determine Lorentz force on a moving charge.

    Arguments:
    q -- value and sign of charge in Coulombs
    v --  array of velocity components (x, y, z)
    B --array of magnetic field components (x,y,z)


    Return
    array force--magnetic force components (x,y,z)
    

    """
    Array = np.cross(q*v, B)
    return np.array([Array[0], Array[1], Array[2] ])

def electron_diffeq(conditions,time):
    """
    This is a function of electron motion
    
    Parameters
    ----------
    
    conditions: list
        A list of the intitial conditions, in the order x,y,z,  x', y', z' , Bx, By, Bz
        
    time: array
        time, not actually used in the calculations below, needed to make the rk4 work without an error though. 
        
    Returns
    -------
    
    Array of x_velocity, y_velocity, x_acceleration, y_acceleration
    
    """
    q = 1.602e-19 #charge of electron in coulombs
    
    M=9.11e-31  #Mass of the electron in kg
    
    x_position=conditions[0]
    
    y_position=conditions[1]
    
    z_position=conditions[2]
    
    x_velocity=conditions[3]
    
    y_velocity=conditions[4]
    
    z_velocity=conditions[5]
    
    Bx = conditions[6]
    
    By = conditions[7]
    
    Bz = conditions[8]
    
    B = np.array([Bx, By, Bz])
    
    velocity = np.array([x_velocity, y_velocity, z_velocity])
    
    x_acceleration= ( np.cross(q*velocity, B) / M)[0]
    
    y_acceleration= ( np.cross(q*velocity, B) / M)[1]
    
    z_acceleration= ( np.cross(q*velocity, B) / M)[2]
    
    return np.array([x_velocity,y_velocity,z_velocity, x_acceleration, y_acceleration, z_acceleration])



def rk4_steps(f, x0, t, dt, **kwargs):
    """This is just a single step in the rk4 function
    
    Parameters
    ----------
    
    f: function
        the differential equation you are integrating over
    
    x0: array
        the initial conditions you are supplying to the differential equation you are integrating
        
    t: float
        the starting time for the step
        
    dt: float
        the time step taken
        
    Returns
    -------
    
    x: array
        the new values after taking a time step
    """
    x = np.copy(x0)
    
    k1 = dt * f(x, t, **kwargs)

    k2 = dt * f(x + 0.5 * k1, t + 0.5 * dt, **kwargs)
    
    k3 = dt * f(x + 0.5 * k2, t + 0.5 * dt, **kwargs)
    
    k4 = dt * f(x + k3, t + dt, **kwargs)
    
    x += (k1 + 2 * k2 + 2 * k3 + k4) / 6 
    return x

def adaptive_rk4(function, x0, t_start, t_end, dt=8.64e4, delta=3.171e-5, max_step_size_increase=2, print_time_step=False):
    """4th order rk method with adaptive step sizes
    
    Parameters
    -----------
    
    function: function
        the differential equation you are integrating over
    
    x0: array
        the initial conditions you are supplying to the differential equation you are integrating
        
    t_start: float
        the starting time 
    
    t_end: float
        the ending time, the difference between t_end and t_start is how long the simulation will run for
        
    dt: float
        The initial time step you are taking, default is 86400 seconds or 1 day
        
    delta: float
        the target error you are want the function to have. The default is 1 kilometer per year or 3.171e-5 m/s
        
    max_step_size_increase: float
        The coefficient factor used to cap how big the time step can grow. The default is 2.
        
    print_time_step: bool
        If True then the function will print out the current time step it is taking, useful for debugging.
        
    Returns
    -------
    
    x_list: array
        The array of all the values calculated by the adaptive rk4 (often positions and velocities)
    """
    
    x = np.copy(x0)     #this is copying the original initial conditions into their own variable
    x_list = x          #this is starting the array we are going to stack with the new updated values
    
    
    while t_start < t_end:
        if print_time_step == True:
            print("current time step is: ", dt/86400 , "days")      #use this to see how the time step changes
        
        x_half = rk4_steps(function, x, t_start, dt / 2)            #this is half a step in the rk method
        
        x1 = rk4_steps(function, x_half, t_start+dt / 2, dt / 2)    #this is the other half for a full step
        
        x2 = rk4_steps(function, x, t_start, dt)                    #this is one large full step
        
        error_x = np.abs( x1[0] - x2[0] ) / 30                      #this calculates the error in the x positions
         
        error_y = np.abs( x1[1] - x2[1] ) / 30                      #this calculates the error in the y positions
        
        total_error = np.sqrt( error_x**2 + error_y**2 )            #this is the total error in both x and y
        
        if total_error <= 1e-12:        #this is so if the total error is near 0 or 0 we don't get a divide by zero
            
            t_start = t_start + dt      #this is updating the time to be the old time plus the time step
            
            dt=2*dt #if the total error is really small (~0), dt would grow too fast. this caps the new dt to 2x
            
            x=x1    #this replaces the old x array with the  the new values
            
            x_list = np.vstack((x_list, x)) #this stacks the new values onto the growing array of values
            
        else:  
            rho=30 * dt * delta / total_error #rho is the ratio used to determine if the error is too big or small
                    
            if rho >= 1: #if rho is >1 that means the error was small and we can update the values and time steps :)
                
                t_start = t_start + dt #updating the time step
                
                dt_new = dt * rho ** (1/4) #this is how the new time step size is determined
                
                if dt_new >=max_step_size_increase*dt: #if the new time step is too big its capped 
                    
                    dt=max_step_size_increase*dt #increasing the time step by the given value (default 2)
                    
                else: #if the updated time step wasnt big, we can change it to whatever new value the equation gave
                    
                    dt = dt_new #updating the time step to the new time step
            
                x=x1 #setting x equal to the 2 combined half steps
                
                x_list = np.vstack((x_list, x)) #stacking the 2 combined half steps (now as x) to the array
        
            else: #if rho <1 that means the error was big so we need to make the time step smaller and repeat the step
                #notice the time and positions are not updated, only the time step is made smaller.
                
                dt = dt * rho ** (1/4) #updating the time step
    
    return x_list #returning the array we have been stacking the values onto


