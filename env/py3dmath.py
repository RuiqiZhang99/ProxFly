"""
@author: mwm

corresponds to the Vec3 class in Common/Math
"""

##UGLY TO PUT EVERYTHING IN __INIT__!

from __future__ import division, print_function #automatically do float division
import numpy as np

class Vec3:
    def __init__(self, *args):
        if len(args) == 0:
            self.x = self.y = self.z = None
        elif len(args) == 1:
            inval = args[0]
            if inval is None:
                self.x = self.y = self.z = None
            elif isinstance(inval,Vec3):
                self.x = inval.x
                self.y = inval.y
                self.z = inval.z
            elif type(inval).__name__ == 'ndarray':
                inval = inval.flatten()
                self.x = inval[0]
                self.y = inval[1]
                self.z = inval[2]
            elif type(inval).__name__ == 'list':
                self.x = inval[0]
                self.y = inval[1]
                self.z = inval[2]
            elif type(inval).__name__ == 'matrix':
                if inval.shape[0] > inval.shape[1]:
                    self.x = inval[0,0]
                    self.y = inval[1,0]
                    self.z = inval[2,0]
                else:
                    self.x = inval[0,0]
                    self.y = inval[0,1]
                    self.z = inval[0,2]
            else:
                raise NameError('Unknown initialisation! (arg type)')
        elif len(args) == 3:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
        else:
            raise NameError('Unknown initialisation! (num args)')


        return
    
    @classmethod
    def from_list(cls, list):
        return cls(list[0], list[1], list[2])

    def norm2(self):
        return (self.x**2+self.y**2+self.z**2)**(1/2)
    
    def norm2Squared(self):
        return self.x**2+self.y**2+self.z**2
    
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def cross(self, other):
        return Vec3([self.y*other.z - self.z*other.y,self.z*other.x - self.x*other.z,self.x*other.y - self.y*other.x])
    
    def to_list(self):
        return [self.x, self.y, self.z]
        
    def to_array(self):
        return np.array([self.to_list()]).T
        
    def to_matrix(self):
        return np.matrix([self.to_list()]).T
        
    def to_unit_vector(self):
        return self/self.norm2()
    
    def to_cross_product_matrix(self):
        return np.matrix([[0,       -self.z,  +self.y],
                          [+self.z,       0,  -self.x], 
                          [-self.y, +self.x,        0]])
        
    def __add__(self, other):
        return Vec3([self.x+other.x,self.y+other.y,self.z+other.z])

    def __sub__(self, other):
        return Vec3([self.x-other.x,self.y-other.y,self.z-other.z])
        
    def __mul__(self, scalar):
        return Vec3([self.x*scalar,self.y*scalar,self.z*scalar])
                
    def __div__(self, scalar):
        return Vec3([self.x/scalar,self.y/scalar,self.z/scalar])
                
    def __rmul__(self, mulVal):
        try:
            #assume you're multiplying a 3x3 matrix:
            if not mulVal.shape==(3,3):
                print("Cannot multiply matrix of shape:", mulVal.shape())
                raise
            
            return Vec3(mulVal*self.to_array())
        except:
            #assume a scalar:
            return Vec3([self.x*mulVal,self.y*mulVal,self.z*mulVal])
        
    def __truediv__ (self, scalar):
        return Vec3([self.x/scalar,self.y/scalar,self.z/scalar])
        
    def __str__(self):
        return '({0}, {1}, {2})'.format(self.x,self.y, self.z)
    
    def __repr__(self):
        return  'Vec3({0},{1},{2})'.format(self.x,self.y, self.z)
        
    def __neg__(self):
        return Vec3([-self.x, -self.y, -self.z])
        
    #square brackets:
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        
        
    #square brackets:
    def __setitem__(self, index, val):
        if index == 0:
            self.x = val
        elif index == 1:
            self.y = val
        elif index == 2:
            self.z = val
                    


class Rotation:   
    def __init__(self, q0, q1, q2, q3):
        self.q=np.array([q0,q1,q2,q3])

    @classmethod
    def from_list(cls, qList):
        return cls(qList[0],qList[1],qList[2],qList[3])
    
    def normalise(self):
        self.q = self.q/np.sqrt(np.dot(self.q,self.q))
        
    def inverse(self):
        return Rotation(self.q[0],-self.q[1],-self.q[2],-self.q[3])

    @classmethod        
    def from_axis_angle(cls, unitVector, angle):
        return Rotation(np.cos(angle/2.),
                        np.sin(angle/2.)*unitVector[0],
                        np.sin(angle/2.)*unitVector[1],
                        np.sin(angle/2.)*unitVector[2])
        
    @classmethod
    def from_rotation_vector(cls, rotVec):
        rotVec = Vec3(rotVec)
        theta = rotVec.norm2()
        if(theta < 4.84813681e-9):
            return Rotation.identity()  # less than one milli arc second :)
        return Rotation.from_axis_angle(rotVec/theta,theta)

    @classmethod
    def from_vector_part_of_quaternion(cls, vec):
        vec = Vec3(vec)
        return Rotation(np.sqrt(1 - vec[0]**2 - vec[1]**2 - vec[2]**2), vec[0], vec[1], vec[2])
        
    
    @classmethod
    def from_euler_YPR(cls, ypr):
        y = ypr[0]
        p = ypr[1]
        r = ypr[2]
        return Rotation(np.cos(0.5*y)*np.cos(0.5*p)*np.cos(0.5*r) + np.sin(0.5*y)*np.sin(0.5*p)*np.sin(0.5*r),
                        np.cos(0.5*y)*np.cos(0.5*p)*np.sin(0.5*r) - np.sin(0.5*y)*np.sin(0.5*p)*np.cos(0.5*r),
                        np.cos(0.5*y)*np.sin(0.5*p)*np.cos(0.5*r) + np.sin(0.5*y)*np.cos(0.5*p)*np.sin(0.5*r),
                        np.sin(0.5*y)*np.cos(0.5*p)*np.cos(0.5*r) - np.cos(0.5*y)*np.sin(0.5*p)*np.sin(0.5*r))


    @classmethod
    def from_rotation_matrix(cls, mat):
        #(Zipfel, P.97-98)
        rotAngle  = np.arccos(0.5*(mat[0,0]+mat[1,1]+mat[2,2])-0.5)
        rotVector = Vec3([mat[2,1]-mat[1,2],mat[0,2]-mat[2,0],mat[1,0]-mat[0,1]])*(1/(2*np.sin(rotAngle)))
        return Rotation.from_axis_angle(rotVector,rotAngle)
    

    @classmethod
    def from_rodrigues_vector(cls, vec):
        vec = Vec3(vec)
        q = (1/np.sqrt(1+vec.norm2Squared()))*np.array([1, vec[0], vec[1], vec[2]])
        return Rotation(q[0], q[1], q[2], q[3])
    

    def to_rodrigues_vector(self):
        """Outputs the corresponding Rodrigues vector, as used e.g. by openCV"""
        return Vec3(np.array([self.q[1],self.q[2],self.q[3]])/self.q[0])
    
    
    def to_vector_part_of_quaternion(self):
        if self.q[0] >= 0:
            return Vec3(self.q[1], self.q[2], self.q[3])
        else:
            return -Vec3(self.q[1], self.q[2], self.q[3])
        

    def to_euler_YPR(self):
        y = np.arctan2(2.0*self.q[1]*self.q[2] + 2.0*self.q[0]*self.q[3], self.q[1]*self.q[1] + self.q[0]*self.q[0] - self.q[3]*self.q[3] - self.q[2]*self.q[2])
        p = -np.arcsin(2.0*self.q[1]*self.q[3] - 2.0*self.q[0]*self.q[2])
        r = np.arctan2(2.0*self.q[2]*self.q[3] + 2.0*self.q[0]*self.q[1], self.q[3]*self.q[3] - self.q[2]*self.q[2] - self.q[1]*self.q[1] + self.q[0]*self.q[0])
        return np.array([y,p,r])
    

    def to_list(self):
        return [self.q[0], self.q[1], self.q[2], self.q[3]]
        

    def to_array(self):
        return np.array([self.to_list()]).T
        

    def to_rotation_vector(self):
        n =  self.to_vector_part_of_quaternion()
        theta = np.arcsin(n.norm2())*2
        if(abs(theta) < 1e-15):
            return Vec3(0,0,0)
        
        return theta*n.to_unit_vector()
    
    
    def to_rotation_matrix(self):
        return self._rotation_matrix()

    
    @classmethod
    def identity(cls):
        return Rotation(1,0,0,0)
        
    def __mul__(self, rhs):
        if(isinstance(rhs,Rotation)):
            #apply successive rotations
            c0 = rhs.q[0]*self.q[0] - rhs.q[1]*self.q[1] - rhs.q[2]*self.q[2] - rhs.q[3]*self.q[3]
            c1 = rhs.q[1]*self.q[0] + rhs.q[0]*self.q[1] + rhs.q[3]*self.q[2] - rhs.q[2]*self.q[3]
            c2 = rhs.q[2]*self.q[0] - rhs.q[3]*self.q[1] + rhs.q[0]*self.q[2] + rhs.q[1]*self.q[3]
            c3 = rhs.q[3]*self.q[0] + rhs.q[2]*self.q[1] - rhs.q[1]*self.q[2] + rhs.q[0]*self.q[3]
            return Rotation(c0,c1,c2,c3)
        else:
            #rotate the vector
            return self.Rotate(rhs)

    
    def Rotate(self, v):
        if(isinstance(v,Vec3)):
            return Vec3(self._rotation_matrix()*v.to_matrix())

        return self._rotation_matrix()*v

    
    def RotateBackwards(self, v):
        vnp = np.matrix([v[0,0],v[1,0],v[2,0]]).T
        vr = self.Inverse()._rotation_matrix()*vnp
        return Vec3.FromNumpyMatrix(vr)
    
    def _rotation_matrix(self):
        r0=self.q[0]*self.q[0]
        r1=self.q[1]*self.q[1]
        r2=self.q[2]*self.q[2]
        r3=self.q[3]*self.q[3]
    
        R = np.matrix(np.zeros([3,3]))
        R[0,0] = r0 + r1 - r2 - r3
        R[0,1] = 2*self.q[1]*self.q[2] - 2*self.q[0]*self.q[3]
        R[0,2] = 2*self.q[1]*self.q[3] + 2*self.q[0]*self.q[2]
    
        R[1,0] = 2*self.q[1]*self.q[2] + 2*self.q[0]*self.q[3]
        R[1,1] = r0 - r1 + r2 - r3
        R[1,2] = 2*self.q[2]*self.q[3] - 2*self.q[0]*self.q[1]
    
        R[2,0] = 2*self.q[1]*self.q[3] - 2*self.q[0]*self.q[2]
        R[2,1] = 2*self.q[2]*self.q[3] + 2*self.q[0]*self.q[1]
        R[2,2] = r0 - r1 - r2 + r3
        return R
    
    def __str__(self):
        return  '{0},{1},{2},{3}'.format(self.q[0],self.q[1], self.q[2], self.q[3])        
#         r=self._rotation_matrix()
#         return '{0}\t{1}\t{2}\n{3}\t{4}\t{5}\n{6}\t{7}\t{8} '.format(r[0,0],r[0,1],r[0,2],r[1,0],r[1,1],r[1,2],r[2,0],r[2,1],r[2,2])

    def __repr__(self):
        return  'Rotation({0},{1},{2},{3})'.format(self.q[0],self.q[1], self.q[2], self.q[3])        
    
    def PrintRotationMatrix(self):
        r=self._rotation_matrix()
        for i in range(0,3):
            print(r[i,0], r[i,1], r[i,2])


